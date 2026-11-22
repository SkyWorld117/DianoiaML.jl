function configured_dense(weights::Matrix{Float32}, activation=None)
    layer = Dense(input_shape=(size(weights, 2),), layer_size=size(weights, 1),
                  activation_function=activation, randomization=false)
    layer.weights .= weights
    return layer
end

@testset "Nested networks train like a flat stack" begin
    make_layers() = (
        configured_dense(reshape(Float32[0.2, -0.4], 2, 1), tanH),
        configured_dense(Float32[0.3 0.1; -0.2 0.6], tanH),
        configured_dense(Float32[0.7 -0.5]),
    )
    layers = make_layers()
    block = Sequential(layers[2], ResNet(layers[3]))
    nested = Sequential(layers[1], block)
    flat = Sequential(make_layers()...)

    @test nested.layers[3] === block
    @test nested.input_shape == nested.output_shape == (1,)
    @test block.input_shape == (2,)
    @test block.output_shape == (1,)

    x = Float32[-1 -0.5 0 0.5 1]
    y = 0.4f0 .* x
    for _ in 1:2
        for model in (nested, flat)
            Adam.fit(; model, input_data=x, output_data=y, loss_function=Mean_Squared_Error,
                      monitor=Absolute, epochs=3, batch=2, shuffle=false, verbose=false)
            model.initialize(model, size(x, 2))
        end
        @test nested.activate(nested, x) ≈ flat.activate(flat, x)
        for (i, layer) in enumerate(layers)
            @test layer.weights ≈ flat.layers[i+1].weights
            @test layer.Sdw ≈ flat.layers[i+1].Sdw
        end
    end
end

@testset "Nested output uses the fused loss gradient" begin
    classifier = configured_dense(reshape(Float32[-1000, 1000], 2, 1), Softmax)
    model = Sequential(Sequential(ResNet(classifier)))
    x = ones(Float32, 1, 1)
    y = reshape(Float32[1, 0], 2, 1)
    SGD.fit(; model, input_data=x, output_data=y, loss_function=Categorical_Cross_Entropy_Loss,
             monitor=Absolute, epochs=1, α=0.1, verbose=false)
    @test vec(classifier.∇biases) == Float32[-1, 1]
    @test vec(classifier.weights) ≈ Float32[-999.9, 999.9] atol=1f-4
end

@testset "Residual stacks propagate training mode and survive a roundtrip" begin
    dropout = Dropout(input_shape=(16,), rate=0.5)
    weights = reshape(Float32.(1:16) ./ 100f0, 1, 16)
    branch = Sequential(dropout, ResNet(configured_dense(Matrix{Float32}(I, 16, 16))))
    model = Sequential(Residual(branch), Sequential(configured_dense(weights)))
    x = ones(Float32, 16, 8)
    model.initialize(model, size(x, 2))
    expected = copy(model.activate(model, x))
    @test expected ≈ weights * (2f0 .* x)
    @test !dropout.training
    model.activate(model, x; training=true)
    @test dropout.training

    mktempdir() do dir
        path = joinpath(dir, "nested.h5")
        save_Sequential(model, path)
        loaded = load_Sequential(path)
        loaded.initialize(loaded, size(x, 2))
        @test loaded.activate(loaded, x) ≈ expected
        @test loaded.layers[2].resnet isa Sequential
        @test loaded.layers[2].resnet.layers[3] isa ResNet
        @test loaded.layers[3] isa Sequential
    end
end

@testset "Composed networks validate topology and shapes" begin
    leaf = configured_dense(ones(Float32, 1, 1))
    block = Sequential(leaf)
    model = Sequential(block)
    @test_throws ArgumentError add_layer!(model, block)
    @test_throws ArgumentError add_layer!(block, model)
    @test_throws ArgumentError add_layer!(model, Sequential())
    @test_throws ArgumentError Sequential(block, Residual(block))
    @test_throws ArgumentError Sequential(block, Sequential(leaf))
    @test length(model.layers) == 2

    @test_throws DimensionMismatch add_layer!(model, Sequential(configured_dense(ones(Float32, 1, 2))))
    @test_throws DimensionMismatch Residual(Sequential(configured_dense(ones(Float32, 2, 1))))
    independent = Sequential(configured_dense(ones(Float32, 1, 1)))
    add_layer!(model, independent)
    @test model.layers[3] === independent
    add_layer!(block, configured_dense(ones(Float32, 2, 1)))
    @test_throws DimensionMismatch model.initialize(model, 1)

    grown = Sequential(block)
    add_layer!(grown, Dense; layer_size=1, activation_function=None)
    @test grown.layers[3].input_shape == (2,)
end

@testset "Genetic optimizer reaches nested parameters" begin
    candidate(weight) = Sequential(Residual(Sequential(ResNet(configured_dense(fill(weight, 1, 1))))))
    elite = candidate(2f0)
    child = candidate(0f0)
    x = Float32[1 2 3]
    GA.fit(; models=[elite, child], input_data=x, output_data=3f0 .* x, monitor=Absolute,
           epochs=1, batch=2, num_copy=1, α=0.0, rng=MersenneTwister(17), verbose=false)
    @test child.layers[2].resnet.layers[2].layers[2].weights == fill(2f0, 1, 1)
end
