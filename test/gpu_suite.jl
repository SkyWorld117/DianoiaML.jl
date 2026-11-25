@testset "DianoiaML GPU ($GPU_NAME)" begin
    seed!(123)
    @testset "Layer values and handwritten gradients" begin
        for activation in (Identity(), ReLU(), Sigmoid(), Softmax(), Tanh(), GELU())
            check_layer(Dense(7, 5; activation), random_tensor(7, 3, 2))
            check_layer(activation, random_tensor(5, 2, 2))
        end
        check_layer(Dense(3, 2), random_tensor(3, 2099))
        for Layer in (Conv2D, TConv2D)
            check_layer(Layer(2, 3; kernel_size=(3, 2), padding=1, stride=(2, 2), activation=Tanh()),
                        random_tensor(6, 5, 2, 3))
        end
        check_layer(Conv2D(32, 64; kernel_size=(3, 3), padding=1), random_tensor(12, 13, 32, 8))
        check_layer(MaxPooling2D((3, 2); stride=(1, 2)), random_tensor(7, 5, 2, 3))
        check_layer(MaxPooling2D((2, 2)), ones(Float32, 3, 3, 2, 2))
        check_layer(UpSampling2D((2, 3); activation=Sigmoid()), random_tensor(3, 2, 2, 2))
        check_layer(LayerNorm(5), random_tensor(5, 3, 2))
        check_layer(Residual(Sequential(Dense(5, 5; activation=GELU()), Dense(5, 5))), random_tensor(5, 4, 2))

        check_layer(Attention(6; heads=2, causal=true), random_tensor(6, 5, 3))
        mask = Bool[1 0; 1 0; 0 0; 1 0; 0 0]
        check_layer(Attention(6; heads=2, context_features=4),
                    (random_tensor(6, 3, 2), random_tensor(4, 5, 2), mask))
        check_layer(TransformerBlock(8; heads=2, hidden_size=12, causal=true),
                    (random_tensor(8, 5, 2), mask))

        projection = Dense(2, 2; activation=Tanh())
        split = Graph(; inputs=(:x,), nodes=(mapped=Node(projection, :x),), outputs=(:x, :mapped))
        graph = Graph(; inputs=(:x,), nodes=(split=Node(split, :x), selected=Node(Select(2), :split),
                              shared=Node(projection, :x), sum=Node(Add(), :selected, :shared)))
        device_model = check_layer(graph, random_tensor(2, 3, 4))
        weights = parameters(device_model)
        @test weights["shared.weights"] === weights["split.mapped.weights"]
        @test state_dict(to_device(CPU(), device_model)) == state_dict(graph)
    end

    @testset "Precision and loss functions" begin
        for T in (Float16, BFloat16, Float64)
            if !DianoiaML.gpu_precision_available(GPU_BACKEND, T)
                @test_throws ArgumentError gpu(Dense(1, 1; precision=T))
                continue
            end
            layer = Dense(3, 2; precision=T, activation=GELU())
            check_layer(layer, T.(random_tensor(3, 3, 2)); atol=0.02, rtol=0.02)
            check_layer(layer, T.(random_tensor(3, 513)); atol=0.02, rtol=0.02)
            spatial = Conv2D(1, 2; kernel_size=(2, 2), precision=T)
            check_layer(spatial, T.(random_tensor(4, 4, 1, 2)); atol=0.02, rtol=0.02)
        end
        @test_throws ArgumentError gpu(Dense(1, 1; precision=BigFloat))

        x = Float32[0.2 0.5; 0.8 0.5]
        y = Float32[0 1; 1 0]
        for loss in (MeanSquaredError(), QuadraticLoss(), AbsoluteLoss(), CategoricalCrossEntropy(),
                     BinaryCrossEntropy(), CrossEntropyWithLogits(), BinaryCrossEntropyWithLogits())
            @test loss(gpu(x), y) ≈ loss(x, y)
            dest = gpu(similar(x))
            check_tensor(loss_gradient!(dest, loss, gpu(x), y), loss_gradient!(similar(x), loss, x, y))
        end
        @test Classification()(gpu(x), y) == Classification()(x, y)
        @test Absolute()(gpu(x), y) ≈ Absolute()(x, y)
    end

    @testset "Batches and optimizers" begin
        x, y = random_tensor(3, 5), random_tensor(2, 5)
        for (a, b) in zip(batches(x, y; batch_size=2, rng=Xoshiro(4)),
                          batches(gpu(x), gpu(y); batch_size=2, rng=Xoshiro(4)))
            @test host(b) == a
        end
        for opt in (SGD(), SGD(; momentum=0.9), Adam(), AdamW(), RMSProp(), AdaBelief())
            model = Dense(3, 2)
            cpu_state, gpu_state = TrainingState(model, opt), TrainingState(gpu(model), opt)
            @test all(slot -> on_test_gpu(slot.gradient), gpu_state.slots)
            for (state, data) in ((cpu_state, (x, y)), (gpu_state, (gpu(x), gpu(y))))
                fit!(state, batches(data...; batch_size=2, shuffle=false); loss=MeanSquaredError(),
                     epochs=2, accumulate=2, clip_norm=1)
            end
            for (name, value) in state_dict(gpu_state.model)
                @test value ≈ state_dict(cpu_state.model)[name] atol=3f-5 rtol=5f-4
            end
        end
    end

    @testset "Dropout and checkpoints" begin
        x = gpu(ones(Float32, 8, 8))
        t = TrainingState(Dropout(1//2), SGD(); seed=18)
        mask = copy(forward!(t, x))
        @test Set(Array(mask)) == Set([0f0, 2f0])
        @test Array(backward!(t, x)) == Array(mask)
        @test Array(predict(t.model, x)) == Array(x)
        model = gpu(Sequential(Dropout(1//4), Dense(8, 2)))
        state = TrainingState(model, AdamW(); seed=20)
        targets = zeros(Float32, 2, 8)
        fit!(state, [(x, targets)]; loss=MeanSquaredError())
        forward!(state, x)
        backward!(state, targets; loss=MeanSquaredError())
        mktempdir() do directory
            file = save_checkpoint(joinpath(directory, "state.h5"), state)
            resumed = TrainingState(gpu(Sequential(Dropout(1//4), Dense(8, 2))), SGD())
            load_checkpoint!(resumed, file)
            @test state_dict(resumed.model) == state_dict(model)
            reference = TrainingState(to_device(CPU(), model), SGD())
            load_checkpoint!(reference, file)
            @test state_dict(reference.model) == state_dict(model)
            for (name, gradient) in gradients(resumed)
                @test Array(gradient) == gradients(reference)[name]
            end
            @test Array(forward!(resumed, x)) == Array(forward!(state, x))
            for current in (state, resumed)
                backward!(current, targets; loss=MeanSquaredError())
                step!(current)
            end
            @test state_dict(resumed.model) == state_dict(model)
        end
    end

    @testset "Parameter-free state follows its inputs" begin
        state = TrainingState(Sequential(MaxPooling2D((2, 2)), Flatten(), Softmax()), SGD())
        x = random_tensor(4, 4, 2, 3)
        expected = predict(state.model, x)
        @test forward!(state, x) == expected
        backward!(state, ones(Float32, size(expected)))
        check_tensor(forward!(state, gpu(x)), expected)
        @test on_test_gpu(backward!(state, gpu(ones(Float32, size(expected)))))
        @test forward!(state, x) == expected
    end

    @testset "GAN and population training" begin
        model = gpu(GAN(Dense(3, 4), Dense(4, 1)))
        noise(rng, count) = randn(rng, Float32, 3, count)
        state = GANState(model; noise, seed=3)
        previous = state_dict(model)
        fit!(state, batches(gpu(random_tensor(4, 6)); batch_size=4))
        @test previous != state_dict(model)
        @test size(generate(model, random_tensor(3, 2))) == (4, 2)

        elite, child = gpu(Dense(1, 1; randomization=false)), gpu(Dense(1, 1; randomization=false))
        elite.weights .= 2f0
        x = gpu(reshape(Float32.(1:4), 1, :))
        fit_population!([elite, child], batches(x, 2f0 .* x; batch_size=3);
                        monitor=Absolute(), keep=1, mutation=0, epochs=1, verbose=false)
        @test state_dict(elite) == state_dict(child)
    end
    include("builtins_gpu.jl")
    synchronize(GPU_BACKEND)
end
