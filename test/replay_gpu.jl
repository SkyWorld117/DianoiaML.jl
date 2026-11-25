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
