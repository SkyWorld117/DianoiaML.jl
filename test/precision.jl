@testset "CPU precision" begin
    for T in (Float16, BFloat16, Float32, Float64, BigFloat)
        @testset "$T" begin
            A = T in (Float16, BFloat16) ? Float32 : T
            model = dense_model(2, 2; precision=T)
            layer = only(model.layers)
            state = TrainingState(model, Adam(); seed=42)
            input = view(Float64[1 2 4; 2 4 8], :, 1:2)
            target = oneHot([1, 1], 2, Dict(1 => 1, 2 => 2); precision=T)
            prediction = forward!(state, input)
            @test eltype(prediction) === eltype(layer.weights) === eltype(target) === T
            @test iszero(prediction)
            derivative = backward!(state, target; loss=CrossEntropyWithLogits())
            @test eltype(derivative) === Float64
            @test iszero(derivative)
            grad = gradients(state)["1.weights"]
            @test eltype(grad) === A
            @test grad == A[-3//4 -3//2; 3//4 3//2]
            step!(state; clip_norm=1)
            @test eltype(state.slots[1].first) === eltype(state.slots[1].second) === A
            logits = predict(model, input)
            @test all(isfinite, logits)
            @test all(logits[1, :] .> logits[2, :])

            set_group!(state, layer; lr=T(1//4))
            forward!(state, input)
            backward!(state, target; loss=CrossEntropyWithLogits())
            mktempdir() do directory
                path = save_checkpoint(joinpath(directory, "model.h5"), state)
                loaded = dense_model(2, 2; precision=T)
                resumed = TrainingState(loaded, SGD())
                load_checkpoint!(resumed, path)
                @test state_dict(loaded) == state_dict(model)
                @test gradients(resumed) == gradients(state)
                @test resumed.slots[1].first == state.slots[1].first
                @test typeof(resumed.slots[1].lr) === typeof(state.slots[1].lr)
                @test resumed.slots[1].lr == state.slots[1].lr
                step!(resumed)
                step!(state)
                @test state_dict(resumed.model) == state_dict(state.model)
            end

            dropout = TrainingState(Dropout(1//2), SGD(); seed=21)
            mask_values = forward!(dropout, ones(T, 8, 8))
            @test eltype(mask_values) === T
            @test Set(mask_values) == Set(T[0, 2])
            @test backward!(dropout, ones(T, 8, 8)) == mask_values

            projection = configured_dense(fill(T(2), 1, 1))
            graph = Graph(; inputs=(:x,), nodes=(left=Node(projection, :x), right=Node(projection, :x),
                                                   sum=Node(Add(), :left, :right)))
            graph_state = TrainingState(graph, SGD())
            @test forward!(graph_state, ones(1, 2)) == fill(T(4), 1, 2)
            @test eltype(graph_state.run.output) === T
            @test backward!(graph_state, ones(1, 2)) == fill(T(4), 1, 2)
            @test gradients(graph_state)["left.weights"] == fill(A(2), 1, 1)
        end
    end
end

@testset "Spatial precision" begin
    x = reshape(Float64.(1:8), 2, 2, 1, 2)
    for Layer in (Conv2D, TConv2D)
        function spatial_model(T)
            layer = Layer(1, 1; kernel_size=(2, 2), padding=1, stride=(2, 1),
                          randomization=false, precision=T)
            layer.filters .= reshape([1, 2, 3, 4], 1, 1, 2, 2)
            return layer
        end
        reference = TrainingState(spatial_model(Float64), SGD())
        expected = copy(forward!(reference, x))
        delta = ones(size(expected))
        expected_input = copy(backward!(reference, delta))
        for T in (Float16, BFloat16, Float32, Float64, BigFloat)
            state = TrainingState(spatial_model(T), SGD())
            prediction = forward!(state, x)
            @test eltype(prediction) === T
            @test prediction == expected
            @test backward!(state, delta) == expected_input
            @test gradients(state) == gradients(reference)
        end
    end

    for T in (Float16, BFloat16, Float64, BigFloat)
        model = Sequential(UpSampling2D((2, 2)), MaxPooling2D((2, 2)), Flatten(),
                           Reshape((2, 2, 1)))
        state = TrainingState(model, SGD())
        prediction = forward!(state, T.(x))
        @test eltype(prediction) === T
        @test prediction == x
        derivative = backward!(state, ones(size(x)))
        @test eltype(derivative) === T
        @test derivative == ones(size(x))
    end
end

@testset "Mixed and arbitrary precision" begin
    @test eltype(Dense(1, 1).weights) === Float32
    setprecision(BigFloat, 384) do
        layer = Dense(1, 1; precision=BigFloat, randomization=false)
        model = Sequential(Residual(layer))
        tiny = BigFloat(2)^(-320)
        layer.weights .= 1 + tiny
        layer.bias .= setprecision(BigFloat, 448) do
            BigFloat(2)^(-414)
        end
        state = TrainingState(model, SGD(; lr=tiny))
        @test only(forward!(state, ones(BigFloat, 1, 1))) == 2 + tiny
        @test only(backward!(state, ones(BigFloat, 1, 1))) == 2 + tiny
        saved = state_dict(model)
        mktempdir() do directory
            path = joinpath(directory, "big.h5")
            setprecision(BigFloat, 256) do
                save_checkpoint(path, state)
                load_checkpoint!(state, path)
                @test state_dict(model) == saved
                @test precision(only(layer.weights)) == 384
                @test precision(only(layer.bias)) == 448
                @test precision(state.optimizer.lr) == 384
            end
        end
    end

    high = configured_dense(Float64[1;;])
    low = configured_dense(Float32[1;;])
    x = fill(1.0 + 2.0^-35, 1, 1)
    @test predict(Sequential(high), x) == x
    @test predict(Sequential(high, low), x) == Float32[1;;]
    mixed = Sequential(low, high)
    state = TrainingState(mixed, SGD())
    @test eltype(forward!(state, ones(Float16, 1, 2))) === Float64
    @test eltype(backward!(state, ones(1, 2))) === Float16
end

@testset "GAN precision" begin
    for T in (Float16, BFloat16, Float64, BigFloat)
        noise(rng, n) = ones(T, 1, n)
        generator = configured_dense(fill(T(1//2), 1, 1))
        discriminator = configured_dense(fill(T(1//4), 1, 1))
        gan = GAN(generator, discriminator)
        state = GANState(gan; noise, generator_optimizer=SGD(; lr=1//10),
                         discriminator_optimizer=SGD(; lr=1//10))
        input = fill(2.0, 1, 2)
        fit!(state, batches(input; batch_size=2))
        @test only(generator.weights) > T(1//2)
        @test eltype(generate(gan, input)) === T
        @test all(values(state_dict(gan))) do array
            eltype(array) === T && all(isfinite, array)
        end
    end
end
