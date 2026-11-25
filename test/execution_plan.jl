@testset "Prepared execution" begin
    shared = Dense(2, 2; activation=Tanh())
    model = Graph(; inputs=(:x,), nodes=(a=Node(shared, :x), b=Node(shared, :a),
                                       skip=Node(Add(), :a, :b)))
    x = (; x=Float32[0 1 2; 1 2 3])
    y = Float32[1 2 3; 2 4 6]
    inference = prepare(model, x)
    first_output = predict(inference, x)
    @test first_output ≈ predict(model, x)
    next = (; x=x.x .+ 0.25f0)
    @test predict(inference, next) === first_output
    @test first_output ≈ predict(model, next)
    tail = (; x=x.x[:, 3:3])
    @test predict(inference, tail) ≈ predict(model, tail)
    @test predict(inference, x) === first_output

    reference = TrainingState(deepcopy(model), AdamW(); seed=42)
    state = TrainingState(model, AdamW(); seed=42)
    plan = prepare(state, x, y; loss=MeanSquaredError())
    data = [(x, y), (next, y .+ 0.5f0), (tail, y[:, 3:3])]
    @test fit!(plan, data; epochs=2, accumulate=2, clip_norm=0.1, verbose=false) === state
    fit!(reference, data; loss=MeanSquaredError(), epochs=2, accumulate=2, clip_norm=0.1, verbose=false)
    @test state_dict(state.model) == state_dict(reference.model)
    @test state.forwards == reference.forwards == 6
    @test state.epochs == reference.epochs == 2
    @test all(slot -> slot.step == 4, state.slots)

    forward!(state, x)
    @test_throws ArgumentError step!(plan, x, y)
    backward!(state, y; loss=MeanSquaredError())
    step!(state)
    @test step!(plan, x, y) === state
    @test state.samples == state.microbatches == 0
end
