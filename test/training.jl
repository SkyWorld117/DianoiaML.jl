mutable struct TrainingProbe
    initialize::Function
    update::Function
    loss::Float64
end

@testset "Batch coverage and adaptive step counts" begin
    x = reshape(Float32.(1:5), 1, :)
    y = 2f0 .* x
    for optimizer in (Adam, AdaBelief, Minibatch_GD, SGD)
        batches = Vector{Vector{Float32}}()
        steps = Int[]
        sizes = Int[]
        monitor = (func=(actual, target) -> 0.0,)
        tracker = TrainingProbe((_, size) -> push!(sizes, size),
            (_, data, targets, loss, monitor, opt, rate, args...) -> begin
                @test targets == 2f0 .* data
                push!(batches, vec(copy(data)))
                isempty(args) || push!(steps, args[1])
            end, 0.0)
        options = optimizer === Minibatch_GD ? (batch=3, mini_batch=2) : (batch=2.0,)
        optimizer.fit(; model=tracker, input_data=x, output_data=y, loss_function=Quadratic_Loss,
                      monitor, epochs=2, shuffle=false, verbose=false, options...)
        @test batches == repeat([Float32[1, 2], Float32[3, 4], Float32[5]], 2)
        @test sizes == [2, 1, 2, 1]
        @test steps == (optimizer in (Adam, AdaBelief) ? collect(1:6) : Int[])
    end

    model = dense_model(1, 1)
    for batch in (0, -1, 1.5)
        @test_throws ArgumentError SGD.fit(; model, input_data=x, output_data=y, loss_function=Quadratic_Loss,
                                            monitor=Absolute, batch, verbose=false)
    end
    @test_throws DimensionMismatch Adam.fit(; model, input_data=x, output_data=y[:, 1:2],
                                             loss_function=Quadratic_Loss, monitor=Absolute, verbose=false)
end

@testset "SGD batch sizes and legacy entry point" begin
    x = ones(Float32, 1, 3)
    y = Float32[1 2 3]
    options = (input_data=x, output_data=y, loss_function=Quadratic_Loss, monitor=Absolute,
               epochs=1, α=0.1, shuffle=false, verbose=false)
    for (batch, expected) in ((1, 0.912f0), (2, 0.78f0), (3, 0.4f0), (6, 0.4f0))
        model = dense_model(1, 1)
        SGD.fit(; model, batch, options...)
        @test only(model.layers[2].weights) ≈ expected
        @test only(model.layers[2].biases) ≈ expected
    end

    legacy = dense_model(1, 1)
    modern = dense_model(1, 1)
    Minibatch_GD.fit(; model=legacy, batch=3, mini_batch=2, options...)
    SGD.fit(; model=modern, batch=2, options...)
    @test legacy.layers[2].weights == modern.layers[2].weights
    @test legacy.layers[2].biases == modern.layers[2].biases
end

@testset "Optimizer equations and repeated fits" begin
    x = ones(Float32, 1, 1)
    y = 0.7f0 .* x
    for optimizer in (Adam, AdaBelief)
        model = dense_model(1, 1)
        layer = model.layers[2]
        w = 0.0
        α, β₁, β₂, ϵ = 0.05, 0.8, 0.9, 1e-8
        for run in 1:2
            m = v = 0.0
            for t in 1:3
                g = 2 * (2w - only(y))
                m = β₁ * m + (1 - β₁) * g
                residual = optimizer === AdaBelief ? g - m : g
                v = β₂ * v + (1 - β₂) * residual^2
                w -= α * (m / (1 - β₁^t)) / (sqrt(v / (1 - β₂^t)) + ϵ)
            end
            optimizer.fit(; model, input_data=x, output_data=y, loss_function=Quadratic_Loss,
                          monitor=Absolute, epochs=3, batch=1, α, β₁, β₂, ϵ, verbose=false)
            @test only(layer.weights) ≈ w atol=2f-7
            @test only(layer.biases) ≈ w atol=2f-7
        end
    end
end

@testset "Convergence" begin
    rng = MersenneTwister(24)
    x = 2f0 .* rand(rng, Float32, 2, 33) .- 1f0
    y = Float32[0.7 -1.2] * x .+ 0.15f0
    for optimizer in (SGD, Adam, AdaBelief)
        model = dense_model(2, 1)
        optimizer.fit(; model, input_data=x, output_data=y, loss_function=Mean_Squared_Error,
                      monitor=Absolute, epochs=60, batch=8, α=0.04, rng=MersenneTwister(59), verbose=false)
        model.initialize(model, size(x, 2))
        @test Absolute.func(model.activate(model, x), y) / size(x, 2) < 0.015
    end

    x = zeros(Float32, 4, 4, 1, 17)
    x[:, 1:2, 1, 1:2:end] .= 1
    x[:, 3:4, 1, 2:2:end] .= 1
    y = oneHot([isodd(i) ? 1 : 2 for i in axes(x, 4)], 2, Dict(1 => 1, 2 => 2))
    model = Sequential()
    model.add_layer(model, Conv2D; input_shape=(4, 4, 1), filter=2, kernel_size=(2, 2),
                    activation_function=ReLU, randomization=false)
    model.layers[2].filters .= 0.2f0
    model.add_layer(model, MaxPooling2D; pool_size=(2, 2), strides=(1, 1))
    model.add_layer(model, Flatten)
    model.add_layer(model, Dense; layer_size=2, activation_function=Softmax, randomization=false)
    Adam.fit(; model, input_data=x, output_data=y, loss_function=Categorical_Cross_Entropy_Loss,
              monitor=Classification, epochs=15, batch=4, α=0.03, rng=MersenneTwister(6), verbose=false)
    model.initialize(model, size(x, 4))
    prediction = model.activate(model, x)
    @test Classification.func(prediction, y) == 0
    @test sum(Categorical_Cross_Entropy_Loss.func(prediction, y)) / size(x, 4) < 0.10
end

@testset "Genetic optimizer with perfect fitness" begin
    elite, candidate = dense_model(1, 1), dense_model(1, 1)
    elite.layers[2].weights .= 2f0
    elite.layers[2].biases .= 1f0
    x = reshape(Float32.(1:5), 1, :)
    y = 2f0 .* x .+ 1f0
    GA.fit(; models=[elite, candidate], input_data=x, output_data=y, monitor=Absolute,
           epochs=2, num_copy=1, mini_batch=2, α=0.0, rng=MersenneTwister(17), verbose=false)
    @test candidate.layers[2].weights == elite.layers[2].weights
    @test candidate.layers[2].biases == elite.layers[2].biases
end
