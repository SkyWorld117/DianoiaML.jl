@testset "Activations and losses" begin
    values = Float32[-1000 3; 1000 -1; 0 2]
    output, gradient = similar(values), similar(values)
    delta = Float32[1 0.3; -2 0.1; 4 -0.8]
    Sigmoid.func!(output, values)
    @test output[:, 1] == Float32[0, 1, 0.5]
    Sigmoid.get_∇biases!(gradient, values, delta)
    @test gradient[:, 1] == Float32[0, 0, 1]
    @test all(isfinite, gradient)

    Softmax.func!(output, values)
    @test output[:, 1] == Float32[0, 1, 0]
    @test all(isapprox.(sum(output; dims=1), 1f0))
    Softmax.get_∇biases!(gradient, values, delta)
    objective() = (Softmax.func!(output, values); dot(output, delta))
    for i in 4:6
        @test gradient[i] ≈ finite_difference!(objective, values, i) atol=2f-4
    end

    prediction = reshape(Float32[0.3, 0.7, 0.2, 0.6, 0.5, 0.8, 0.4, 0.1], 2, 2, 1, 2)
    target = reshape(Float32[0, 1, 1, 0, 0, 1, 0, 1], size(prediction))
    response = similar(prediction)
    for loss in (Absolute_Loss, Quadratic_Loss, Mean_Squared_Error, Binary_Cross_Entropy_Loss, Categorical_Cross_Entropy_Loss)
        @test size(loss.func(prediction, target)) == size(prediction)
        loss.prop!(response, prediction, target)
        loss_value() = sum(loss.func(prediction, target))
        @test response[8] ≈ finite_difference!(loss_value, prediction, 8) atol=8f-4
    end
    @test sum(Mean_Squared_Error.func(prediction, target)) ≈ sum(abs2, prediction .- target) / 4
    Absolute_Loss.prop!(response, target, target)
    @test iszero(response)

    @test Classification.func(Float32[0.1 0.8], Float32[0 1]) == 0
    @test Classification.func(Float32[0.7 0.4], Float32[0 1]) == 2
    @test oneHot(reshape([0, 1, 1, 0], 2, 2), 2, Dict(0 => 1, 1 => 2)) == Float32[1 0 0 1; 0 1 1 0]
end

@testset "Fused cross-entropy gradients" begin
    x = ones(Float32, 1, 1)
    for activation in (Softmax, Softmax_CEL)
        model = dense_model(1, 2, activation)
        layer = model.layers[2]
        layer.weights[:, 1] .= Float32[-1000, 1000]
        model.initialize(model, 1)
        target = reshape(Float32[0.75, 0.5], 2, 1)
        model.update(model, x, target, Categorical_Cross_Entropy_Loss, Absolute, "SGD", 0.1)
        @test vec(layer.∇biases) ≈ Float32[-0.75, 0.75]
        @test vec(layer.weights) ≈ Float32[-999.925, 999.925] atol=1f-4
    end

    model = dense_model(1, 1, Sigmoid)
    model.layers[2].weights[1] = -1000f0
    model.initialize(model, 1)
    model.update(model, x, x, Binary_Cross_Entropy_Loss, Absolute, "SGD", 0.1)
    @test model.layers[2].∇biases[1] == -1f0
    @test model.layers[2].weights[1] ≈ -999.9f0 atol=1f-4
end

function reference_convolution(layer, input)
    result = similar(layer.output)
    sx, sy = layer.strides
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
        result[i, j, f, b] = layer.biases[f]
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
        layer = Layer(; input_shape=(4, 5, 2), filter=3, kernel_size=(2, 3), strides=(2, 1),
                      padding=1, activation_function=None, randomization=false)
        layer.initialize(layer, 2)
        layer.filters .= rand(rng, Float32, size(layer.filters)) .- 0.5f0
        layer.biases .= rand(rng, Float32, size(layer.biases))
        x = rand(rng, Float32, 4, 5, 2, 2)
        delta = rand(rng, Float32, size(layer.output)) .- 0.5f0
        layer.activate(layer, x)
        @test maximum(abs, layer.output .- reference_convolution(layer, x)) < 5f-7
        layer.update(layer, "Minibatch_GD", x, delta, 0.0, ())
        objective() = (layer.activate(layer, x); dot(layer.output, delta))
        for i in (1, 5, 22, 58, 80)
            @test layer.δ[i] ≈ finite_difference!(objective, x, i) atol=6f-4
        end
        for i in (1, 6, 17, 24, 36)
            @test layer.∇weights[i] ≈ finite_difference!(objective, layer.filters, i) / 2 atol=4f-4
        end
        for i in eachindex(layer.biases)
            @test layer.∇biases_c[i] ≈ finite_difference!(objective, layer.biases, i) / 2 atol=4f-4
        end
    end

    pooling = MaxPooling2D(input_shape=(3, 3, 2), pool_size=(2, 2), strides=(1, 1))
    pooling.initialize(pooling, 3)
    x = zeros(Float32, 3, 3, 2, 3)
    x[2, 2, :, :] .= 7f0
    pooling.activate(pooling, x)
    @test all(==(7f0), pooling.output)
    pooling.update(pooling, "SGD", x, ones(Float32, 2, 2, 2, 3), 0.1, ())
    @test pooling.δ[2, 2, :, :] == fill(4f0, 2, 3)
    @test sum(pooling.δ) == 24f0

    upsample = UpSampling2D(input_shape=(2, 3, 1), size=(2, 3), activation_function=None)
    upsample.initialize(upsample, 2)
    x = reshape(Float32.(1:12), 2, 3, 1, 2)
    upsample.activate(upsample, x)
    @test upsample.output == repeat(x; inner=(2, 3, 1, 1))
    upsample.update(upsample, "SGD", x, ones(Float32, 4, 9, 1, 2), 0.1, ())
    @test all(==(6f0), upsample.δ)
end

@testset "Convolution on a larger batch" begin
    layer = Conv2D(input_shape=(32, 32, 8), filter=32, kernel_size=(3, 3),
                   padding=1, activation_function=None, randomization=false)
    layer.filters .= 1f0
    layer.initialize(layer, 8)
    x = ones(Float32, 32, 32, 8, 8)
    delta = ones(Float32, 32, 32, 32, 8)
    for _ in 1:3
        layer.activate(layer, x)
        layer.update(layer, "Minibatch_GD", x, delta, 0.0, ())
        @test all(==(9f0 * 8), layer.output[2:31, 2:31, :, :])
        @test all(==(4f0 * 8), layer.output[1, 1, :, :])
        @test all(==(9f0 * 32), layer.δ[2:31, 2:31, :, :])
        @test all(==(4f0 * 32), layer.δ[1, 1, :, :])
        @test all(==(32f0^2), layer.∇weights[:, :, 2, 2])
        @test all(==(31f0^2), layer.∇weights[:, :, 1, 1])
        @test all(==(32f0^2), layer.∇biases_c)
    end
end

@testset "Dense and residual backpropagation" begin
    branch = ResNet()
    branch.add_layer(branch, Dense; input_shape=(2,), layer_size=2, activation_function=None, randomization=false)
    layer = branch.layers[2]
    layer.weights .= Float32[1 2; 3 4]
    residual = Residual(input_shape=(2,), resnet=branch)
    residual.initialize(residual, 2)
    x = Float32[1 3; 2 4]
    delta = Float32[-2 0.1; 0.5 -1]
    residual.activate(residual, x)
    @test residual.output ≈ layer.weights * x + x
    residual.update(residual, "SGD", x, delta, 0.0, ())
    @test residual.δ ≈ delta + transpose(layer.weights) * delta
    @test layer.weight_gradient ≈ delta * transpose(x) / 2
    @test layer.bias_gradient ≈ vec(sum(delta; dims=2) / 2)
end
