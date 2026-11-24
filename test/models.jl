struct ScaleLayer <: AbstractLayer
    factor::Float32
end
DianoiaML.output_shape(::ScaleLayer, input) = input
DianoiaML.forward!(layer::ScaleLayer, ::Nothing, input, mode) = layer.factor .* input
DianoiaML.backward!(layer::ScaleLayer, ::Nothing, input, delta; compute_parameters=true) = layer.factor .* delta

@testset "Custom layer contract" begin
    dense = configured_dense(Float32[1;;])
    model = Sequential(dense, ScaleLayer(2f0))
    x = ones(Float32, 1, 2)
    @test predict(model, x) == fill(2f0, 1, 2)
    fit!(TrainingState(model, SGD(; lr=0.1)), [(x, zero(x))]; loss=QuadraticLoss())
    @test only(dense.weights) ≈ 0.2f0
    @test only(dense.bias) ≈ -0.8f0
    @test_throws DimensionMismatch predict(dense, ones(Float32, 1))
end

@testset "Dropout buffers and resizing" begin
    model = Sequential(Dropout(0.5))
    state = TrainingState(model, SGD(); seed=14)
    x = ones(Float32, 8, 16)
    @test predict(model, x) == x
    trained = forward!(state, x)
    @test Set(trained) == Set(Float32[0, 2])
    @test predict(model, x) == x
    @test backward!(state, x) == trained
    short = x[:, 1:3]
    @test size(forward!(state, short)) == (8, 3)
    @test size(backward!(state, short)) == (8, 3)
    @test_throws ArgumentError Dropout(1.0)
end

@testset "GAN alternating gradients and checkpoint" begin
    for nested in (false, true)
        generator = configured_dense(Float32[0.5;;])
        discriminator = configured_dense(Float32[0.3;;])
        model = nested ? GAN(Sequential(generator), Sequential(discriminator)) : GAN(generator, discriminator)
        noise(rng, n) = ones(Float32, 1, n)
        rate = 0.1
        state = GANState(model; noise, generator_optimizer=SGD(; lr=rate),
                         discriminator_optimizer=SGD(; lr=rate), seed=30)
        real = fill(2f0, 1, 2)
        @test generate(model, ones(Float32, 1, 2)) == fill(0.5f0, 1, 2)
        sigmoid(x) = 1 / (1 + exp(-x))
        real_delta = sigmoid(0.3 * 2) - 1
        d_weight = 0.3 - rate * real_delta * 2
        d_bias = -rate * real_delta
        fake_delta = sigmoid(d_weight * 0.5 + d_bias)
        d_weight -= rate * fake_delta * 0.5
        d_bias -= rate * fake_delta
        g_delta = (sigmoid(d_weight * 0.5 + d_bias) - 1) * d_weight
        data = batches(real; batch_size=2, shuffle=false)
        fit!(state, data)
        @test only(discriminator.weights) ≈ d_weight atol=1f-7
        @test only(discriminator.bias) ≈ d_bias atol=1f-7
        @test only(generator.weights) ≈ 0.5 - rate * g_delta atol=1f-7
        @test only(generator.bias) ≈ -rate * g_delta atol=1f-7
        @test state.discriminator.slots[1].step == 2
        @test state.generator.slots[1].step == 1

        mktempdir() do directory
            path = save_checkpoint(joinpath(directory, "gan.h5"), state)
            loaded = GANState(deepcopy(model); noise)
            load_checkpoint!(loaded, path)
            @test loaded.steps == state.steps
            fit!(loaded, data)
            fit!(state, data)
            @test state_dict(loaded.generator.model) == state_dict(state.generator.model)
            @test state_dict(loaded.discriminator.model) == state_dict(state.discriminator.model)
        end
    end
end
