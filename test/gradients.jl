@testset "Activations and losses" begin
    values = Float32[-1000 3; 1000 -1; 0 2]
    delta = Float32[1 0.3; -2 0.1; 4 -0.8]
    state = TrainingState(Sigmoid(), SGD())
    @test forward!(state, values)[:, 1] == Float32[0, 1, 0.5]
    gradient = backward!(state, delta)
    @test gradient[:, 1] == Float32[0, 0, 1]
    @test all(isfinite, gradient)

    state = TrainingState(Softmax(), SGD())
    output = forward!(state, values)
    @test output[:, 1] == Float32[0, 1, 0]
    @test all(isapprox.(sum(output; dims=1), 1f0))
    gradient = backward!(state, delta)
    objective() = dot(predict(Softmax(), values), delta)
    for i in 4:6
        @test gradient[i] ≈ finite_difference!(objective, values, i) atol=2f-4
    end

    prediction = reshape(Float32[0.3, 0.7, 0.2, 0.6, 0.5, 0.8, 0.4, 0.1], 2, 2, 1, 2)
    target = reshape(Float32[0, 1, 1, 0, 0, 1, 0, 1], size(prediction))
    response = similar(prediction)
    for loss in (AbsoluteLoss(), QuadraticLoss(), MeanSquaredError(), BinaryCrossEntropy(),
                 CategoricalCrossEntropy(), CrossEntropyWithLogits(), BinaryCrossEntropyWithLogits())
        @test isfinite(loss_value(loss, prediction, target))
        loss_gradient!(response, loss, prediction, target)
        @test response[8] ≈ finite_difference!(() -> loss_value(loss, prediction, target), prediction, 8) atol=8f-4
    end
    @test MeanSquaredError()(prediction, target) ≈ sum(abs2, prediction .- target) / 4
    loss_gradient!(response, AbsoluteLoss(), target, target)
    @test iszero(response)

    @test Classification()(Float32[0.1 0.8], Float32[0 1]) == 0
    @test Classification()(Float32[0.7 0.4], Float32[0 1]) == 2
    @test oneHot(reshape([0, 1, 1, 0], 2, 2), 2, Dict(0 => 1, 1 => 2)) == Float32[1 0 0 1; 0 1 1 0]
end

@testset "Explicit losses from logits" begin
    x = ones(Float32, 1, 1)
    layer = configured_dense(reshape(Float32[-1000, 1000], 2, 1))
    state = TrainingState(Sequential(Sequential(layer)), SGD(; lr=0.1))
    logits = forward!(state, x)
    loss = CrossEntropyWithLogits()
    target = reshape(Float32[0.75, 0.5], 2, 1)
    @test loss(logits, target) ≈ 1500
    backward!(state, target; loss)
    @test gradients(state)["1.1.bias"] ≈ Float32[-0.75, 0.75]
    step!(state)
    @test vec(layer.weights) ≈ Float32[-999.925, 999.925] atol=1e-4

    layer = configured_dense(Float32[-1000;;])
    state = TrainingState(layer, SGD(; lr=0.1))
    loss = BinaryCrossEntropyWithLogits()
    @test loss(forward!(state, x), x) == 1000
    backward!(state, x; loss)
    @test only(gradients(state)["bias"]) == -1
    step!(state)
    @test only(layer.weights) ≈ -999.9f0 atol=1e-4
end

function reference_convolution(layer, input, output)
    result = similar(output)
    sx, sy = layer.stride
    p = layer.padding
    expanded = input
    if layer isa TConv2D
        expanded = zeros(Float32, (size(input, 1) - 1) * sx + 2p + 1,
                         (size(input, 2) - 1) * sy + 2p + 1, size(input, 3), size(input, 4))
        expanded[p+1:sx:size(expanded, 1)-p, p+1:sy:size(expanded, 2)-p, :, :] .= input
        sx = sy = 1
        p = 0
    end
    for b in axes(result, 4), f in axes(result, 3), j in axes(result, 2), i in axes(result, 1)
        result[i, j, f, b] = layer.bias[f]
        for v in axes(layer.filters, 4), u in axes(layer.filters, 3), c in axes(layer.filters, 2)
            x, y = (i - 1) * sx + u - p, (j - 1) * sy + v - p
            if checkbounds(Bool, expanded, x, y, c, b)
                result[i, j, f, b] += expanded[x, y, c, b] * layer.filters[f, c, u, v]
            end
        end
    end
    return result
end

@testset "Spatial gradients" begin
    rng = MersenneTwister(21)
    for Layer in (Conv2D, TConv2D)
        layer = Layer(2, 3; kernel_size=(2, 3), stride=(2, 1), padding=1, randomization=false)
        layer.filters .= rand(rng, Float32, size(layer.filters)) .- 0.5f0
        layer.bias .= rand(rng, Float32, size(layer.bias))
        x = rand(rng, Float32, 4, 5, 2, 2)
        state = TrainingState(layer, SGD())
        output = forward!(state, x)
        delta = rand(rng, Float32, size(output)) .- 0.5f0
        @test maximum(abs, output .- reference_convolution(layer, x, output)) < 5f-7
        dx = backward!(state, delta)
        grad = gradients(state)
        objective() = dot(predict(layer, x), delta)
        for i in (1, 5, 22, 58, 80)
            @test dx[i] ≈ finite_difference!(objective, x, i) atol=6f-4
        end
        for i in (1, 6, 17, 24, 36)
            @test grad["filters"][i] ≈ finite_difference!(objective, layer.filters, i) / 2 atol=4f-4
        end
        for i in eachindex(layer.bias)
            @test grad["bias"][i] ≈ finite_difference!(objective, layer.bias, i) / 2 atol=4f-4
        end
    end

    pool = TrainingState(MaxPooling2D((2, 2); stride=(1, 1)), SGD())
    x = zeros(Float32, 3, 3, 2, 3)
    x[2, 2, :, :] .= 7f0
    @test all(==(7f0), forward!(pool, x))
    derivative = backward!(pool, ones(Float32, 2, 2, 2, 3))
    @test derivative[2, 2, :, :] == fill(4f0, 2, 3)
    @test sum(derivative) == 24f0

    upsample = TrainingState(UpSampling2D((2, 3)), SGD())
    x = reshape(Float32.(1:12), 2, 3, 1, 2)
    @test forward!(upsample, x) == repeat(x; inner=(2, 3, 1, 1))
    @test all(==(6f0), backward!(upsample, ones(Float32, 4, 9, 1, 2)))
end

@testset "Max pooling window order and batch splitting" begin
    function check_pool(T, input_shape, pool_size, strides, batch)
        layer = MaxPooling2D(pool_size; stride=strides)
        state = TrainingState(layer, SGD())
        rng = MersenneTwister(12)
        choices = T[-Inf, -2, -0.0, 0.0, 1, 1, 2, Inf, NaN]
        input = rand(rng, choices, input_shape..., batch)
        output = forward!(state, input)
        expected = similar(output)
        indices = similar(state.run.cache.indices)
        linear = LinearIndices(input)
        sx, sy = strides
        px, py = pool_size
        for b in axes(expected, 4), c in axes(expected, 3), j in axes(expected, 2), i in axes(expected, 1)
            x, y = (i - 1) * sx, (j - 1) * sy
            value, index = findmax(view(input, x + 1:x + px, y + 1:y + py, c, b))
            expected[i, j, c, b] = value
            indices[i, j, c, b] = linear[x + index[1], y + index[2], c, b]
        end
        @test eltype(output) === T
        @test isequal(output, expected)
        @test state.run.cache.indices == indices

        delta = rand(rng, T[-2, -1, 0, 1, 2], size(expected))
        input_gradient = zeros(T, size(input))
        for i in eachindex(delta)
            input_gradient[indices[i]] += delta[i]
        end
        @test backward!(state, delta) == input_gradient
    end

    for T in (Float16, BFloat16, Float32, Float64, BigFloat)
        check_pool(T, (3, 5, 2), (2, 3), (1, 1), 3)
        check_pool(T, (8, 9, 2), (1, 2), (2, 3), 5)
    end
    check_pool(Float32, (26, 26, 4), (2, 2), (2, 2), 128)
    check_pool(Float32, (26, 26, 4), (3, 3), (1, 1), 56)
end

@testset "Convolution on a larger batch" begin
    layer = Conv2D(8, 32; kernel_size=(3, 3), padding=1, randomization=false)
    layer.filters .= 1
    state = TrainingState(layer, SGD())
    x = ones(Float32, 32, 32, 8, 8)
    delta = ones(Float32, 32, 32, 32, 8)
    for _ in 1:3
        output = forward!(state, x)
        dx = backward!(state, delta)
        grad = gradients(state)
        @test all(==(9f0 * 8), output[2:31, 2:31, :, :])
        @test all(==(4f0 * 8), output[1, 1, :, :])
        @test all(==(9f0 * 32), dx[2:31, 2:31, :, :])
        @test all(==(4f0 * 32), dx[1, 1, :, :])
        @test all(==(32f0^2), grad["filters"][:, :, 2, 2])
        @test all(==(31f0^2), grad["filters"][:, :, 1, 1])
        @test all(==(32f0^2), grad["bias"])
        zero_grad!(state)
    end
end

@testset "Dense and residual backpropagation" begin
    layer = configured_dense(Float32[1 2; 3 4])
    state = TrainingState(Residual(layer), SGD())
    x = Float32[1 3; 2 4]
    delta = Float32[-2 0.1; 0.5 -1]
    @test forward!(state, x) ≈ layer.weights * x + x
    @test backward!(state, delta) ≈ delta + transpose(layer.weights) * delta
    @test gradients(state)["branch.weights"] ≈ delta * transpose(x) / 2
    @test gradients(state)["branch.bias"] ≈ vec(sum(delta; dims=2) / 2)
end
