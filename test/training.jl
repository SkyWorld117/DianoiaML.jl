@testset "Batch coverage and optimizer steps" begin
    x = reshape(Float32.(1:5), 1, :)
    y = 2f0 .* x
    ordered = batches(x, y; batch_size=2, shuffle=false)
    @test length(ordered) == 3
    for optimizer in (Adam(), AdaBelief(), SGD())
        seen = Vector{Float32}[]
        state = TrainingState(dense_model(1, 1), optimizer)
        monitor(actual, target) = begin
            push!(seen, vec(copy(target)))
            0.0
        end
        fit!(state, ordered; loss=QuadraticLoss(), monitor, epochs=2, verbose=false)
        @test seen == repeat([Float32[2, 4], Float32[6, 8], Float32[10]], 2)
        @test all(slot -> slot.step == 6, state.slots)
        @test state.epochs == 2
    end
    shuffled = batches((features=x, mask=trues(1, 5)), y; batch_size=2, rng=MersenneTwister(2))
    @test sort(vcat([vec(copy(input.features)) for (input, target) in shuffled]...)) == vec(x)
    for (input, target) in shuffled
        @test target == 2 .* input.features
        @test all(input.mask)
    end
    for count in (0, -1)
        @test_throws ArgumentError batches(x, y; batch_size=count)
    end
    @test_throws DimensionMismatch batches(x, y[:, 1:2]; batch_size=1)
end

@testset "SGD batch sizes" begin
    x = ones(Float32, 1, 3)
    y = Float32[1 2 3]
    for (batch, expected) in ((1, 0.912f0), (2, 0.78f0), (3, 0.4f0), (6, 0.4f0))
        model = dense_model(1, 1)
        state = TrainingState(model, SGD(; lr=0.1))
        fit!(state, batches(x, y; batch_size=batch, shuffle=false); loss=QuadraticLoss())
        @test only(model.layers[1].weights) ≈ expected
        @test only(model.layers[1].bias) ≈ expected
    end
end

@testset "Adaptive optimizer equations and repeated fits" begin
    x = ones(Float32, 1, 1)
    y = 0.7f0 .* x
    for optimizer in (Adam, AdaBelief)
        model = dense_model(1, 1)
        layer = only(model.layers)
        w = m = v = 0.0
        lr, beta1, beta2, epsilon = 0.05, 0.8, 0.9, 1e-8
        state = TrainingState(model, optimizer(; lr, beta1, beta2, epsilon))
        for run in 1:2
            for t in (3run-2):3run
                g = 2 * (2w - only(y))
                m = beta1 * m + (1 - beta1) * g
                residual = optimizer === AdaBelief ? g - m : g
                v = beta2 * v + (1 - beta2) * residual^2
                w -= lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + epsilon)
            end
            fit!(state, [(x, y)]; loss=QuadraticLoss(), epochs=3)
            @test only(layer.weights) ≈ w atol=2f-7
            @test only(layer.bias) ≈ w atol=2f-7
            @test state.slots[1].step == 3run
        end
    end
end

@testset "Convergence" begin
    rng = MersenneTwister(24)
    x = 2f0 .* rand(rng, Float32, 2, 33) .- 1f0
    y = Float32[0.7 -1.2] * x .+ 0.15f0
    for optimizer in (SGD, Adam, AdaBelief)
        model = dense_model(2, 1)
        state = TrainingState(model, optimizer(; lr=0.04))
        fit!(state, batches(x, y; batch_size=8, rng=MersenneTwister(59));
             loss=MeanSquaredError(), epochs=60)
        @test Absolute()(predict(model, x), y) / size(x, 2) < 0.015
    end

    x = zeros(Float32, 4, 4, 1, 17)
    x[:, 1:2, 1, 1:2:end] .= 1
    x[:, 3:4, 1, 2:2:end] .= 1
    y = oneHot([isodd(i) ? 1 : 2 for i in axes(x, 4)], 2, Dict(1 => 1, 2 => 2))
    conv = Conv2D(1, 2; kernel_size=(2, 2), activation=ReLU(), randomization=false)
    conv.filters .= 0.2f0
    model = Sequential(conv, MaxPooling2D((2, 2); stride=(1, 1)), Flatten(),
                       Dense(8, 2; randomization=false); input_shape=(4, 4, 1))
    state = TrainingState(model, Adam(; lr=0.03))
    loss = CrossEntropyWithLogits()
    fit!(state, batches(x, y; batch_size=4, rng=MersenneTwister(6)); loss, epochs=15)
    prediction = predict(model, x)
    @test Classification()(prediction, y) == 0
    @test loss(prediction, y) / size(x, 4) < 0.10
end

@testset "Genetic optimizer with perfect fitness" begin
    elite, candidate = configured_dense(Float32[2;;]), configured_dense(Float32[0;;])
    elite.bias .= 1
    x = reshape(Float32.(1:5), 1, :)
    y = 2f0 .* x .+ 1f0
    fit_population!([elite, candidate], batches(x, y; batch_size=2);
                    monitor=Absolute(), epochs=2, keep=1, mutation=0, rng=MersenneTwister(17), verbose=false)
    @test candidate.weights == elite.weights
    @test candidate.bias == elite.bias
end
