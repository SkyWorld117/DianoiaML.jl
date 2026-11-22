@testset "GELU and LayerNorm gradients" begin
    x = Float64[-3 -0.8 0.2; 0.7 1.4 3]
    dy = Float64[0.5 0.4 -0.3; 0.1 -0.2 0.9]
    output, delta = similar(x), similar(x)
    GELU.get_∇biases!(delta, x, dy)
    gelu_objective() = dot(GELU.func!(output, x), dy)
    @test gelu_objective() ≈ sum(x .* (1 .+ tanh.(sqrt(2 / π) .* (x .+ 0.044715 .* x.^3))) .* dy) / 2
    for i in eachindex(x)
        @test delta[i] ≈ finite_difference!(gelu_objective, x, i; h=1e-6) atol=1e-8
    end
    @test GELU.func!(zeros(2), [-Inf, Inf]) == [0.0, Inf]

    rng = MersenneTwister(4)
    layer = LayerNorm(; input_shape=(3, 2), precision=Float64)
    features, upstream = randn(rng, 3, 4, 2), randn(rng, 3, 4, 2)
    normalized = activate!(layer, features)
    @test size(normalized) == size(features)
    @test maximum(abs, sum(normalized; dims=1)) < 1e-12
    layer.scale .= [0.5, -0.25, 1.2]
    layer.bias .= [0.1, 0.3, -0.7]
    objective() = dot(activate!(layer, features), upstream)
    objective()
    input_gradient = copy(backward!(layer, features, upstream))
    for i in (1, 4, 11, 20)
        @test input_gradient[i] ≈ finite_difference!(objective, features, i; h=1e-6) atol=1e-8
    end
    for (value, gradient) in parameter_gradients(layer), i in eachindex(value)
        @test gradient[i] ≈ finite_difference!(objective, value, i; h=1e-6) / size(features, 3) atol=1e-8
    end
end

@testset "Multi-head causal attention" begin
    layer = Attention(; input_shape=(4, 3), heads=2, causal=true, precision=Float64,
                      randomization=false)
    for projection in (layer.query, layer.key, layer.value, layer.projection)
        projection.weights .= Matrix{Float64}(I, 4, 4)
    end
    input = reshape(sin.(Float64.(1:24)), 4, 3, 2)
    expected = similar(input)
    for b in 1:2, head in (1:2, 3:4), token in 1:3
        values = input[head, 1:token, b]
        weights = exp.(transpose(values) * input[head, token, b] / sqrt(2))
        expected[head, token, b] = values * (weights / sum(weights))
    end
    @test activate!(layer, input) ≈ expected
    changed = copy(input)
    changed[:, 3, :] .+= 12
    @test activate!(layer, changed)[:, 1:2, :] ≈ expected[:, 1:2, :]

    upstream = reshape(cos.(Float64.(1:24)), size(input))
    objective() = dot(activate!(layer, input), upstream)
    objective()
    derivative = copy(backward!(layer, input, upstream))
    for i in (1, 4, 9, 23)
        @test derivative[i] ≈ finite_difference!(objective, input, i; h=1e-6) atol=1e-8
    end
    for projection in (layer.query, layer.key, layer.value, layer.projection),
        (value, gradient) in parameter_gradients(projection)
        i = lastindex(value)
        @test gradient[i] ≈ finite_difference!(objective, value, i; h=1e-6) / 2 atol=1e-8
    end
    @test_throws ArgumentError Attention(; input_shape=(4, 3), heads=3)
end

@testset "Cross-attention with padding" begin
    rng = MersenneTwister(2)
    layer = Attention(; input_shape=(4, 3), context_shape=(3, 5), heads=2,
                      padding_mask=true, precision=Float64)
    graph = Graph(((4, 3), (3, 5), (5,)), Node(layer, 1, 2, 3))
    query, context = randn(rng, 4, 3, 2), randn(rng, 3, 5, 2)
    mask = falses(5, 2)
    mask[[1, 3, 4], 1] .= true
    output = copy(activate!(graph, (query, context, mask)))
    @test iszero(output[:, :, 2])
    changed = copy(context)
    changed[:, [2, 5], :] .+= 50
    @test activate!(graph, (query, changed, mask)) == output

    state = TrainingState(graph, Optimizer(:SGD))
    upstream = randn(rng, size(query))
    forward!(state, (query, context, mask))
    dq, dc, dm = backward!(state, upstream)
    @test dm === nothing
    @test iszero(dq[:, :, 2]) && iszero(dc[:, :, 2])
    @test iszero(dc[:, [2, 5], :])
    objective() = dot(activate!(graph, (query, context, mask)), upstream)
    for i in (1, 4, 9)
        @test dq[i] ≈ finite_difference!(objective, query, i; h=1e-6) atol=1e-8
    end
    for i in (1, 7, 11)
        @test dc[i] ≈ finite_difference!(objective, context, i; h=1e-6) atol=1e-8
    end
    @test_throws DimensionMismatch activate!(layer, (query, context, trues(4, 2)))
    @test_throws ArgumentError activate!(layer, (query, context, ones(Int, 5, 2)))
end

@testset "Transformer blocks compose, train, and save" begin
    seed!(7)
    model = Sequential(; precision=Float64)
    add_layer!(model, TransformerBlock; input_shape=(4, 3), heads=2, causal=true, hidden_size=8)
    add_layer!(model, TransformerBlock; heads=2, hidden_size=8)
    add_layer!(model, Dense; layer_size=2, activation_function=None)
    input = reshape(sin.(Float64.(1:24)), 4, 3, 2)
    target = zeros(Float64, 2, 3, 2)
    state = TrainingState(model, Optimizer(:Adam; α=0.005); seed=2)
    loss() = sum(abs2, forward!(state, input; training=false))
    initial = loss()
    fit!(state, [(input, target)]; loss_function=Mean_Squared_Error, epochs=35, verbose=false)
    @test loss() < initial / 5
    @test size(forward!(state, input[:, 1:2, :]; training=false)) == (2, 2, 2)

    mktempdir() do directory
        path = joinpath(directory, "transformer.h5")
        save_Sequential(model, path)
        loaded = load_Sequential(path)
        @test state_dict(loaded) == state_dict(model)
        @test activate!(loaded, input) ≈ forward!(state, input; training=false)
    end

    masked() = Sequential(TransformerBlock(; input_shape=(4, 3), heads=2, padding_mask=true,
                                          causal=true, dropout=0.25))
    state = TrainingState(masked(), Optimizer(:Adam); seed=12)
    mask = trues(3, 2)
    features = Float32.(input)
    fit!(state, [((features, mask), zero(features))]; loss_function=Mean_Squared_Error, verbose=false)
    mktempdir() do directory
        path = joinpath(directory, "masked.h5")
        save_checkpoint(path, state)
        resumed = TrainingState(masked(), Optimizer(:SGD))
        load_checkpoint!(resumed, path)
        @test state_dict(resumed.model) == state_dict(state.model)
        @test forward!(resumed, (features, mask)) == forward!(state, (features, mask))
        @test backward!(resumed, features) == backward!(state, features)
        step!(resumed)
        step!(state)
        @test state_dict(resumed.model) == state_dict(state.model)
    end
end

@testset "Transformer precision" begin
    for T in (Float16, BFloat16, Float32, Float64, BigFloat)
        block = TransformerBlock(; input_shape=(4, 2), heads=2, precision=T)
        state = TrainingState(block, Optimizer(:Adam; precision=T))
        x = reshape(sin.(1.0:16.0), 4, 2, 2)
        @test eltype(forward!(state, x)) === T
        @test eltype(backward!(state, zero(x); loss_function=Mean_Squared_Error)) === T
        step!(state)
        @test all(arr -> eltype(arr) === T && all(isfinite, arr), values(parameters(block)))
    end
end
