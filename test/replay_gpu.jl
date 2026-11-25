@testset "Native replay ($GPU_NAME)" begin
    seed!(37)
    architecture = Sequential(Dense(11, 9; activation=GELU()), Dense(9, 2))
    model = gpu(architecture)
    x, y = gpu(random_tensor(11, 3, 4)), gpu(random_tensor(2, 3, 4))
    inference = prepare(model, x)
    for _ in 1:4
        x .+= 0.1f0
        check_tensor(predict(inference, x), Array(predict(model, x)))
    end
    @test inference.replay !== nothing && inference.replay.launch !== nothing

    loss = MeanSquaredError()
    reference = TrainingState(gpu(architecture), AdamW(); seed=4)
    state = TrainingState(model, AdamW(); seed=4)
    plan = prepare(state, x, y; loss)
    for _ in 1:4
        x .-= 0.2f0
        y .+= 0.05f0
        step!(plan, x, y)
        forward!(reference, x)
        backward!(reference, y; loss)
        step!(reference)
    end
    @test plan.replay !== nothing && plan.replay.launch !== nothing
    data = ((x, y), (x .+ 1f0, y .- 0.5f0), (x, y))
    fit!(plan, data; accumulate=2, clip_norm=1, verbose=false)
    fit!(reference, data; loss, accumulate=2, clip_norm=1, verbose=false)
    @test state.forwards == reference.forwards == 7
    @test state.microbatches == state.samples == 0
    for (name, weight) in parameters(state.model)
        check_tensor(weight, Array(parameters(reference.model)[name]))
    end
end

if GPU_NAME === :metal
    @testset "Metal replay preserves bias reduction order" begin
        values = gpu(random_tensor(32, 26 * 26 * 125))
        expected = gpu(zeros(Float32, 32))
        actual = similar(expected)
        DianoiaML.column_sum!(expected, values, 125f0)
        launch = DianoiaML.capture_replay(GPU_BACKEND, nothing) do
            DianoiaML.column_sum!(actual, values, 125f0)
        end
        @test launch !== nothing
        @test launch()
        @test Array(actual) == Array(expected)
    end

    @testset "Metal convolution replay" begin
        for T in (Float32, Float16)
            architecture = Sequential(
                Conv2D(2, 4; kernel_size=(3, 3), activation=ReLU(), precision=T),
                BatchNorm2D(4; precision=T), MaxPooling2D((2, 2)), Flatten(),
                Dense(100, 3; precision=T); input_shape=(13, 13, 2),
            )
            model = gpu(architecture)
            x = gpu(random_tensor(13, 13, 2, 7))
            y = gpu(oneHot(mod.(1:7, 3), 3, Dict(i => i+1 for i in 0:2)))
            partial = (gpu(random_tensor(13, 13, 2, 3)), gpu(Float32[1 0 0; 0 1 0; 0 0 1]))
            loss = CrossEntropyWithLogits()
            state = TrainingState(model, AdamW())
            reference = TrainingState(gpu(architecture), AdamW())
            plan = prepare(state, x, y; loss)
            inference = prepare(model, x)
            atol = T === Float16 ? 2f-3 : 3f-5

            data = ((x, y), (x, y), (x, y), partial, (x, y), (x, y))
            for (i, batch) in enumerate(data)
                x .+= 0.1f0
                fit!(plan, (batch,); monitor=Classification(), verbose=false)
                fit!(reference, (batch,); loss, verbose=false)
                check_tensor(predict(inference, x), Array(predict(reference.model, x)); atol)
                i == 3 && @test plan.replay.launch !== nothing
                i == 4 && @test plan.replay.launch === nothing
            end
            @test plan.replay.launch !== nothing
            @test inference.replay.launch !== nothing
            @test state.forwards == reference.forwards == 6
            expected = state_dict(reference.model)
            for (name, value) in state_dict(model)
                @test value ≈ expected[name] atol=atol rtol=5f-4
            end
        end
    end

    @testset "Metal graph replay with branching and views" begin
        architecture = Graph(; inputs=(:x, :context), nodes=(
            position=Node(PositionalEmbedding(3, 5), :x),
            combined=Node(Add(), :position, :context),
            joined=Node(Concatenate(1), :combined, :x),
            permuted=Node(Permute((2, 1)), :joined),
            left=Node(Dense(4, 2; activation=Sigmoid()), :permuted),
            right=Node(Dense(4, 2; activation=Sigmoid()), :permuted),
            difference=Node(Subtract(), :left, :right),
        ), outputs=(:difference,))
        model = gpu(architecture)
        x = (x=gpu(random_tensor(3, 4, 2)), context=gpu(random_tensor(3, 4, 2)))
        y = gpu(ones(Float32, 2, 6, 2))
        loss = BinaryCrossEntropyWithLogits()
        inference = prepare(model, x)
        state, reference = TrainingState(model, Adam()), TrainingState(gpu(architecture), Adam())
        plan = prepare(state, x, y; loss)
        for _ in 1:4
            x.x .+= 0.1f0
            x.context .*= 0.9f0
            step!(plan, x, y)
            fit!(reference, ((x, y),); loss, verbose=false)
            check_tensor(predict(inference, x), Array(predict(reference.model, x)))
        end
        @test plan.replay.launch !== nothing
        @test inference.replay.launch !== nothing
        for (name, weight) in parameters(state.model)
            check_tensor(weight, Array(parameters(reference.model)[name]))
        end
    end

    @testset "Metal masked Transformer replay" begin
        architecture = TransformerBlock(8; heads=2, causal=true, dropout=0)
        tokens = gpu(random_tensor(8, 5, 3))
        mask = gpu(trues(5, 3))
        x, y = (tokens, mask), gpu(random_tensor(8, 5, 3))
        loss = MeanSquaredError()
        state = TrainingState(gpu(architecture), Adam())
        reference = TrainingState(gpu(architecture), Adam())
        plan = prepare(state, x, y; loss)
        for i in 1:4
            tokens .+= 0.05f0
            mask .= gpu([row <= 6 - i for row in 1:5, _ in 1:3])
            step!(plan, x, y)
            fit!(reference, ((x, y),); loss, verbose=false)
        end
        @test plan.replay.launch !== nothing
        for (name, weight) in parameters(state.model)
            check_tensor(weight, Array(parameters(reference.model)[name]))
        end
    end
end

if GPU_NAME === :cuda
    @testset "Portable kernel recording" begin
        model = gpu(Sequential(Dense(11, 9; activation=Tanh()), Dense(9, 2)))
        x, y = gpu(random_tensor(11, 3, 4)), gpu(random_tensor(2, 3, 4))
        @test DianoiaML.recordable_workload(model, x, MeanSquaredError())
        @test !DianoiaML.recordable_workload(model, x, CrossEntropyWithLogits())
        @test !DianoiaML.recordable_workload(model, ones(Float32, 11, 513), MeanSquaredError())

        state = TrainingState(model, AdamW())
        DianoiaML.train_batch!(state, x, y, MeanSquaredError())
        expected = Dict(name => Array(gradient) for (name, gradient) in gradients(state))
        zero_grad!(state)
        recorded = DianoiaML.record_launches() do
            DianoiaML.train_batch!(state, x, y, MeanSquaredError())
        end
        @test length(recorded) == 17
        @test all(gradient -> all(iszero, Array(gradient)), values(gradients(state)))
        for launch in recorded
            DianoiaML.KI.launch(launch.kernel, launch.groups, launch.items, launch.args)
        end
        for (name, gradient) in gradients(state)
            check_tensor(gradient, expected[name])
        end
    end
end
