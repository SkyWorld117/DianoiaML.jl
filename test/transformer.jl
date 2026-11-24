@testset "GELU and LayerNorm gradients" begin
    x = Float64[-3 -0.8 0.2; 0.7 1.4 3]
    dy = Float64[0.5 0.4 -0.3; 0.1 -0.2 0.9]
    gelu = GELU()
    state = TrainingState(gelu, SGD())
    forward!(state, x)
    delta = backward!(state, dy)
    gelu_objective() = dot(predict(gelu, x), dy)
    @test gelu_objective() ≈ sum(x .* (1 .+ tanh.(sqrt(2 / π) .* (x .+ 0.044715 .* x.^3))) .* dy) / 2
    for i in eachindex(x)
        @test delta[i] ≈ finite_difference!(gelu_objective, x, i; h=1e-6) atol=1e-8
    end
    @test predict(gelu, [-Inf, Inf]) == [0.0, Inf]

    rng = MersenneTwister(4)
    layer = LayerNorm(3; precision=Float64)
    features, upstream = randn(rng, 3, 4, 2), randn(rng, 3, 4, 2)
    normalized = predict(layer, features)
    @test size(normalized) == size(features)
    @test maximum(abs, sum(normalized; dims=1)) < 1e-12
    layer.scale .= [0.5, -0.25, 1.2]
    layer.bias .= [0.1, 0.3, -0.7]
    objective() = dot(predict(layer, features), upstream)
    state = TrainingState(layer, SGD())
    forward!(state, features)
    input_gradient = backward!(state, upstream)
    for i in (1, 4, 11, 20)
        @test input_gradient[i] ≈ finite_difference!(objective, features, i; h=1e-6) atol=1e-8
    end
    grad = gradients(state)
    for (name, value) in parameters(layer), i in eachindex(value)
        @test grad[name][i] ≈ finite_difference!(objective, value, i; h=1e-6) / size(features, 3) atol=1e-8
    end
end

@testset "Multi-head causal attention" begin
    layer = Attention(4; heads=2, causal=true, precision=Float64, randomization=false)
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
    @test predict(layer, input) ≈ expected
    changed = copy(input)
    changed[:, 3, :] .+= 12
    @test predict(layer, changed)[:, 1:2, :] ≈ expected[:, 1:2, :]

    upstream = reshape(cos.(Float64.(1:24)), size(input))
    objective() = dot(predict(layer, input), upstream)
    state = TrainingState(layer, SGD())
    forward!(state, input)
    dx = backward!(state, upstream)
    for i in (1, 4, 9, 23)
        @test dx[i] ≈ finite_difference!(objective, input, i; h=1e-6) atol=1e-8
    end
    grad = gradients(state)
    for (name, value) in parameters(layer)
        i = lastindex(value)
        @test grad[name][i] ≈ finite_difference!(objective, value, i; h=1e-6) / 2 atol=1e-8
    end
    @test_throws ArgumentError Attention(4; heads=3)
end

@testset "Cross-attention with padding" begin
    rng = MersenneTwister(2)
    layer = Attention(4; context_features=3, heads=2, precision=Float64)
    graph = Graph(; inputs=(:query, :context, :mask),
                  nodes=(attention=Node(layer, :query, :context, :mask),))
    @test shape(graph) == (4, nothing)
    @test shape(graph, ((4, 3), (3, 5), (5,))) == (4, 3)
    @test_throws DimensionMismatch shape(graph, ((4, 3), (4, 5), (5,)))
    @test_throws DimensionMismatch shape(graph, ((4, 3), (3, 5), (3,)))
    query, context = randn(rng, 4, 3, 2), randn(rng, 3, 5, 2)
    mask = falses(5, 2)
    mask[[1, 3, 4], 1] .= true
    output = predict(graph, (query, context, mask))
    @test iszero(output[:, :, 2])
    changed = copy(context)
    changed[:, [2, 5], :] .+= 50
    @test predict(graph, (query, changed, mask)) == output

    state = TrainingState(graph, SGD())
    upstream = randn(rng, size(query))
    forward!(state, (query, context, mask))
    dq, dc, dm = backward!(state, upstream)
    @test dm === nothing
    @test iszero(dq[:, :, 2]) && iszero(dc[:, :, 2])
    @test iszero(dc[:, [2, 5], :])
    objective() = dot(predict(graph, (query, context, mask)), upstream)
    for i in (1, 4, 9)
        @test dq[i] ≈ finite_difference!(objective, query, i; h=1e-6) atol=1e-8
    end
    for i in (1, 7, 11)
        @test dc[i] ≈ finite_difference!(objective, context, i; h=1e-6) atol=1e-8
    end
    @test_throws DimensionMismatch predict(layer, (query, context, trues(4, 2)))
    @test_throws ArgumentError predict(layer, (query, context, ones(Int, 5, 2)))
end

@testset "Transformer blocks compose, train, and save" begin
    seed!(7)
    model = Sequential(TransformerBlock(4; heads=2, causal=true, hidden_size=8, precision=Float64),
                       TransformerBlock(4; heads=2, hidden_size=8, precision=Float64),
                       Dense(4, 2; precision=Float64))
    input = reshape(sin.(Float64.(1:24)), 4, 3, 2)
    target = zeros(Float64, 2, 3, 2)
    state = TrainingState(model, Adam(; lr=0.005); seed=2)
    error() = sum(abs2, predict(model, input))
    initial = error()
    fit!(state, [(input, target)]; loss=MeanSquaredError(), epochs=35)
    @test error() < initial / 5
    @test size(predict(model, input[:, 1:2, :])) == (2, 2, 2)

    function masked()
        block = TransformerBlock(4; heads=2, causal=true, dropout=0.25)
        return Graph(; inputs=(:tokens, :key_mask),
                     nodes=(first=Node(block, :tokens, :key_mask), second=Node(block, :first, :key_mask)))
    end
    state = TrainingState(masked(), Adam(); seed=12)
    mask = trues(3, 2)
    features = Float32.(input)
    fit!(state, [((features, mask), zero(features))]; loss=MeanSquaredError())
    @test parameters(state.model)["first.attention.query.weights"] === parameters(state.model)["second.attention.query.weights"]
    mktempdir() do directory
        path = save_checkpoint(joinpath(directory, "masked.h5"), state)
        resumed = TrainingState(masked(), SGD())
        load_checkpoint!(resumed, path)
        @test state_dict(resumed.model) == state_dict(state.model)
        @test forward!(resumed, (features, mask)) == forward!(state, (features, mask))
        @test backward!(resumed, features) == backward!(state, features)
        step!(resumed)
        step!(state)
        @test state_dict(resumed.model) == state_dict(state.model)
    end

    block = TransformerBlock(4; heads=2)
    @test predict(block, features) == predict(block, (features, mask))
end

@testset "Transformer precision" begin
    for T in (Float16, BFloat16, Float32, Float64, BigFloat)
        block = TransformerBlock(4; heads=2, precision=T)
        state = TrainingState(block, Adam())
        x = reshape(T.(sin.(1.0:16.0)), 4, 2, 2)
        @test eltype(forward!(state, x)) === T
        @test eltype(backward!(state, zero(x); loss=MeanSquaredError())) === T
        step!(state)
        @test all(arr -> eltype(arr) === T && all(isfinite, arr), values(parameters(block)))
    end
end
