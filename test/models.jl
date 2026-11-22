mutable struct ScaleLayer{I,A,U}
    input_shape::Tuple{Int}
    output_shape::Tuple{Int}
    initialize::I
    activate::A
    update::U
    output::Matrix{Float32}
    δ::Matrix{Float32}
end

function ScaleLayer(; input_shape)
    initialize = (layer, batch) -> begin
        layer.output = zeros(Float32, layer.output_shape..., batch)
        layer.δ = similar(layer.output)
    end
    activate = (layer, input) -> begin
        layer.output .= 2f0 .* input
        return nothing
    end
    update = (layer, optimizer, input, delta, rate, parameters) -> (layer.δ .= 2f0 .* delta)
    return ScaleLayer(input_shape, input_shape, initialize, activate, update, zeros(Float32, 0, 0), zeros(Float32, 0, 0))
end

struct ConstantNoise
    value::Float32
end

(noise::ConstantNoise)() = fill(noise.value, 1, 1)

@testset "Layer methods and the custom layer protocol" begin
    model = dense_model(1, 1)
    dense = model.layers[2]
    dense.weights .= 1f0
    model.add_layer(model, ScaleLayer)
    @test hasproperty(model, :activate)
    @test hasproperty(dense, :save_layer)
    @test model.layers[3] isa ScaleLayer

    x = ones(Float32, 1, 2)
    y = zero(x)
    model.initialize(model, 2)
    @test model.activate(model, x) == fill(2f0, 1, 2)
    SGD.fit(; model, input_data=x, output_data=y, loss_function=Quadratic_Loss, monitor=Absolute,
             batch=2, epochs=1, α=0.1, verbose=false)
    @test only(dense.weights) ≈ 0.2f0
    @test only(dense.biases) ≈ -0.8f0
    @test_throws MethodError dense.activate(dense, ones(Float32, 1))
end

@testset "Dropout and reinitialization" begin
    model = Sequential()
    model.add_layer(model, Dropout; input_shape=(8,), rate=0.5)
    model.initialize(model, 16)
    layer = model.layers[2]
    x = ones(Float32, 8, 16)
    @test model.activate(model, x) == x
    seed!(14)
    training_output = model.activate(model, x; training=true)
    @test Set(training_output) == Set(Float32[0, 2])
    layer.update(layer, "SGD", x, x, 0.0, ())
    @test layer.δ == training_output
    @test model.activate(model, x) == x
    @test_throws ArgumentError Dropout(input_shape=(1,), rate=1.0)

    model.add_layer(model, Dense; layer_size=2, activation_function=None, randomization=false)
    model.initialize(model, 3)
    @test size(model.activate(model, x[:, 1:3])) == (2, 3)
    @test size(model.layers[end].δ) == (2, 3)
end

@testset "GAN alternating gradients" begin
    noise = ConstantNoise(1f0)
    model = GAN(noise)
    @test (@inferred model.noise_generator()) == ones(Float32, 1, 1)
    model.add_Glayer(model, Dense; input_shape=(1,), layer_size=1, activation_function=None, randomization=false)
    model.add_Dlayer(model, Dense; layer_size=1, activation_function=Sigmoid, randomization=false)
    generator, discriminator = model.layers[2:3]
    generator.weights[1] = 0.5f0
    discriminator.weights[1] = 0.3f0
    model.initialize(model, 2)
    fake = model.activate_Generator(model)
    real = fill(2f0, 1, 2)
    model.activate_Discriminator(model, real)
    @test generator.output === fake
    @test generator.output == fill(0.5f0, 1, 2)

    rate = 0.1
    sigmoid(x) = 1 / (1 + exp(-x))
    real_delta = sigmoid(0.3 * 2) - 1
    d_weight = 0.3 - rate * real_delta * 2
    d_bias = -rate * real_delta
    fake_delta = sigmoid(d_weight * 0.5 + d_bias)
    d_weight -= rate * fake_delta * 0.5
    d_bias -= rate * fake_delta
    g_delta = (sigmoid(d_weight * 0.5 + d_bias) - 1) * d_weight
    model.update(model, real, real, Binary_Cross_Entropy_Loss, Absolute, "SGD", rate)
    @test only(discriminator.weights) ≈ d_weight atol=1f-7
    @test only(discriminator.biases) ≈ d_bias atol=1f-7
    @test only(generator.weights) ≈ 0.5 - rate * g_delta atol=1f-7
    @test only(generator.biases) ≈ -rate * g_delta atol=1f-7

    mktempdir() do path
        filename = joinpath(path, "gan.h5")
        save_GAN(model, filename)
        loaded = load_GAN(filename, noise)
        loaded.initialize(loaded, 2)
        @test loaded.activate(loaded) ≈ model.activate(model)
    end
end

@testset "HDF5 model roundtrip with nested layers" begin
    branch = ResNet()
    branch.add_layer(branch, Dense; input_shape=(2,), layer_size=2, activation_function=tanH, randomization=false)
    branch.layers[2].weights .= Float32[0.1 0.2; 0.3 0.4]
    branch.layers[2].biases .= Float32[0.5, 0.6]
    model = dense_model(2, 2)
    model.layers[2].weights .= Float32[1 0; 0 1]
    model.add_layer(model, Residual; resnet=branch)
    model.add_layer(model, Constructive; shape=(1, 2, 1))
    model.add_layer(model, Flatten)
    model.add_layer(model, Dense; layer_size=2, activation_function=Softmax_CEL, randomization=false)
    model.layers[end].weights .= Float32[0.1 0.2; 0.3 0.4]
    x = Float32[1 3; 2 4]
    model.initialize(model, 2)

    mktempdir() do path
        filename = joinpath(path, "sequential.h5")
        save_Sequential(model, filename)
        loaded = load_Sequential(filename)
        loaded.initialize(loaded, 2)
        @test loaded.activate(loaded, x) ≈ model.activate(model, x)
        @test loaded.layers[3].resnet.layers[2].weights == branch.layers[2].weights
    end
end
