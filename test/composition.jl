@testset "Nested networks train like a flat stack" begin
    make_layers() = (
        configured_dense(reshape(Float32[0.2, -0.4], 2, 1), Tanh()),
        configured_dense(Float32[0.3 0.1; -0.2 0.6], Tanh()),
        configured_dense(Float32[0.7 -0.5]),
    )
    layers = make_layers()
    block = Sequential(layers[2], Sequential(layers[3]))
    nested = Sequential(layers[1], block)
    flat = Sequential(make_layers()...)
    @test nested.layers[2] === block
    @test shape(nested, (1,)) == (1,)
    @test shape(block, (2, 7)) == (1, 7)

    x = Float32[-1 -0.5 0 0.5 1]
    y = 0.4f0 .* x
    states = map(model -> TrainingState(model, Adam()), (nested, flat))
    for _ in 1:2
        for state in states
            fit!(state, batches(x, y; batch_size=2, shuffle=false); loss=MeanSquaredError(), epochs=3)
        end
        @test predict(nested, x) ≈ predict(flat, x)
        for (i, layer) in enumerate(layers)
            @test layer.weights ≈ flat.layers[i].weights
            @test states[1].slots[2i-1].second ≈ states[2].slots[2i-1].second
        end
    end
end

@testset "Residuals wrap layers and graphs" begin
    weights = reshape(Float32.(1:16) ./ 100f0, 1, 16)
    identity = configured_dense(Matrix{Float32}(I, 16, 16))
    branch = Graph(; inputs=(:input,), nodes=(dropped=Node(Dropout(0.5), :input),
                                             projection=Node(identity, :dropped)))
    model = Sequential(Residual(branch), configured_dense(weights))
    x = ones(Float32, 16, 8)
    @test predict(model, x) ≈ weights * (2f0 .* x)
    state = TrainingState(model, SGD(); seed=7)
    @test forward!(state, x) != predict(model, x)
    @test size(backward!(state, ones(Float32, 1, 8))) == size(x)
end

@testset "Composition checks dimensions at construction" begin
    @test_throws ArgumentError Sequential()
    @test_throws DimensionMismatch Sequential(Dense(2, 3), Dense(4, 1))
    @test_throws DimensionMismatch Residual(Sequential(Dense(1, 2)))
    @test_throws DimensionMismatch Sequential(Conv2D(1, 2; kernel_size=(2, 2)), Flatten(), Dense(4, 1);
                                                input_shape=(4, 4, 1))
    @test_throws ArgumentError Graph(; inputs=(:x,), nodes=(x=Node(Dense(1, 1), :x),))
    @test_throws ArgumentError Graph(; inputs=(:x,), nodes=(skip=Node(Dense(1, 1), :later),
                                                              later=Node(Dense(1, 1), :x)))
    @test_throws ArgumentError Graph(; inputs=(:x,), nodes=(one=Node(Dense(1, 1), :x),
                                                              two=Node(Dense(1, 1), :x)))
    @test_throws DimensionMismatch Graph(; inputs=(:x,), input_shapes=(x=(4, nothing),),
                                          nodes=(projection=Node(Dense(3, 1), :x),))
    model = Sequential(Dense(2, 3), Flatten(), Dense(9, 1))
    @test size(predict(model, ones(Float32, 2, 3, 2))) == (1, 2)
    @test predict(Reshape((2, 2)), reshape(Float32.(1:8), 4, 2)) == reshape(Float32.(1:8), 2, 2, 2)
end

@testset "Genetic optimizer reaches nested parameters" begin
    candidate(weight) = Sequential(Residual(Sequential(configured_dense(fill(weight, 1, 1)))))
    elite = candidate(2f0)
    child = candidate(0f0)
    x = Float32[1 2 3]
    fit_population!([elite, child], batches(x, 3f0 .* x; batch_size=2); monitor=Absolute(),
                    keep=1, mutation=0, rng=MersenneTwister(17), verbose=false)
    @test parameters(child)["1.branch.1.weights"] == fill(2f0, 1, 1)
end
