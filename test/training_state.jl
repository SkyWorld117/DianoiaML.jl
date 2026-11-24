@testset "Separate backward, optimizer groups, and accumulation" begin
    layer = configured_dense(ones(Float32, 1, 1))
    state = TrainingState(Sequential(layer), SGD(; lr=0.1))
    x = Float32[1 2 3]
    y = zero(x)
    forward!(state, x[:, 1:1])
    @test backward!(state, y[:, 1:1]; loss=QuadraticLoss()) == Float32[2;;]
    forward!(state, x[:, 2:3])
    backward!(state, y[:, 2:3]; loss=QuadraticLoss())
    @test layer.weights == ones(Float32, 1, 1)
    @test only(gradients(state)["1.weights"]) ≈ 28f0 / 3
    @test only(gradients(state)["1.bias"]) ≈ 4f0

    whole = configured_dense(ones(Float32, 1, 1))
    full = TrainingState(Sequential(whole), SGD(; lr=0.1))
    forward!(full, x)
    backward!(full, y; loss=QuadraticLoss())
    step!(state; clip_norm=2)
    step!(full; clip_norm=2)
    @test layer.weights ≈ whole.weights
    @test layer.bias ≈ whole.bias
    @test hypot(only(layer.weights) - 1, only(layer.bias)) ≈ 0.2

    set_group!(state, layer.bias; trainable=false)
    set_group!(state, layer.weights; lr=0.0)
    before = state_dict(state.model)
    forward!(state, x)
    backward!(state, y; loss=QuadraticLoss())
    step!(state)
    @test state_dict(state.model) == before
end

function tied_graph(; tied=true)
    first = configured_dense(fill(2f0, 1, 1))
    second = tied ? first : configured_dense(fill(2f0, 1, 1))
    return Graph(; inputs=(:a, :b), nodes=(left=Node(first, :a), right=Node(second, :b),
                                          sum=Node(Add(), :left, :right)))
end

@testset "Named graph nodes share parameters and sum fan-out gradients" begin
    graph = tied_graph()
    state = TrainingState(graph, SGD(; lr=0.2))
    params = parameters(graph)
    @test params["left.weights"] === params["right.weights"]
    @test length(gradients(state)) == 2
    set_group!(state, params["left.bias"]; trainable=false)
    a, b = Float32[1 2], Float32[3 4]
    @test forward!(state, (b=b, a=a)) == Float32[8 12]
    @test_throws DimensionMismatch predict(graph, (a=a, c=b))
    @test state.run.children.left.output !== state.run.children.right.output
    @test backward!(state, ones(Float32, 1, 2)) == (b=fill(2f0, 1, 2), a=fill(2f0, 1, 2))
    @test only(gradients(state)["left.weights"]) == 5f0
    step!(state)
    @test only(params["left.weights"]) ≈ 1f0
    @test only(params["right.bias"]) == 0f0

    inner = Graph(; inputs=(:x,), nodes=(two=Node(configured_dense(Float32[2;;]), :x),
                                         three=Node(configured_dense(Float32[3;;]), :x)),
                  outputs=(:two, :three))
    outer = Graph(; inputs=(:x,), nodes=(inner=Node(inner, :x), second=Node(Select(2), :inner),
                                         sum=Node(Add(), :x, :second)), outputs=(:sum, :inner))
    state = TrainingState(outer, SGD())
    x = Float32[1 2]
    @test forward!(state, x) == (4f0 .* x, (2f0 .* x, 3f0 .* x))
    @test backward!(state, (one.(x), (one.(x), nothing))) == fill(6f0, 1, 2)
    @test_throws ArgumentError Graph(; inputs=(:x,), nodes=(;))
end

@testset "Prediction and independent training states preserve pending gradients" begin
    layer = configured_dense(ones(Float32, 1, 1))
    model = Sequential(layer)
    a, b = TrainingState(model, SGD()), TrainingState(model, SGD())
    x, y = ones(Float32, 1, 1), zeros(Float32, 1, 1)
    @test forward!(a, x) == x
    @test_throws ArgumentError forward!(a, x)
    @test predict(model, 10f0 .* x) == 10f0 .* x
    @test forward!(b, 2f0 .* x) == 2f0 .* x
    backward!(a, y; loss=QuadraticLoss())
    backward!(b, y; loss=QuadraticLoss())
    @test only(gradients(a)["1.weights"]) == 2
    @test only(gradients(b)["1.weights"]) == 8
end

struct MaskLayer <: AbstractLayer end
DianoiaML.output_shape(::MaskLayer, input) = first(input)
DianoiaML.forward!(::MaskLayer, ::Nothing, input, mode) = input[1] .* input[2]
DianoiaML.backward!(::MaskLayer, ::Nothing, input, delta; compute_parameters=true) =
    (delta .* input[2], nothing)

@testset "Iterable batches, masks, and variable token lengths" begin
    projection = Dense(1, 1; randomization=false)
    graph = Graph(; inputs=(:x, :mask), nodes=(projection=Node(projection, :x),
                                               masked=Node(MaskLayer(), :projection, :mask)))
    state = TrainingState(graph, SGD(; lr=0.05))
    x = reshape(Float64.(1:6), 1, 3, 2)
    batch(tokens) = ((view(x, :, 1:tokens, :), trues(1, tokens, 2)), Float32.(view(x, :, 1:tokens, :)))
    fit!(state, () -> (batch(tokens) for tokens in (2, 3)); loss=QuadraticLoss(),
         epochs=2, accumulate=2)
    @test state.epochs == 2
    @test state.slots[1].step == 2
    @test size(predict(graph, first(batch(2)))) == (1, 2, 2)
    @test size(predict(graph, first(batch(3)))) == (1, 3, 2)
end

@testset "Checkpoint resumes optimizers, partial batches, and dropout" begin
    checkpoint_model() = Sequential(Dropout(0.5), configured_dense(Float32[0.2 -0.3]))
    state = TrainingState(checkpoint_model(), Adam(; lr=0.02); seed=19)
    set_group!(state, parameters(state.model)["2.bias"]; lr=0.005)
    x = Float32[1 2 3; 2 1 0]
    y = Float32[0.2 0.3 0.4]
    loss = QuadraticLoss()
    fit!(state, [(x, y)]; loss)
    forward!(state, x[:, 1:1])
    backward!(state, y[:, 1:1]; loss)

    mktempdir() do folder
        path = joinpath(folder, "training.h5")
        save_checkpoint(path, state)
        save_checkpoint(path, state)
        resumed = TrainingState(checkpoint_model(), SGD(); seed=0)
        load_checkpoint!(resumed, path)
        @test state_dict(resumed.model) == state_dict(state.model)
        @test gradients(resumed) == gradients(state)
        @test resumed.samples == resumed.microbatches == resumed.epochs == 1
        @test resumed.optimizer isa Adam
        @test resumed.slots[2].lr == 0.005
        @test forward!(resumed, x[:, 2:3]) == forward!(state, x[:, 2:3])
        for training in (resumed, state)
            backward!(training, y[:, 2:3]; loss)
            step!(training)
        end
        @test state_dict(resumed.model) == state_dict(state.model)
        @test resumed.slots[1].first == state.slots[1].first
        @test resumed.slots[1].step == state.slots[1].step == 2

        tied = TrainingState(tied_graph(), Adam())
        save_checkpoint(path, tied)
        untied = TrainingState(tied_graph(; tied=false), Adam())
        before = state_dict(untied.model)
        @test_throws ArgumentError load_checkpoint!(untied, path)
        @test state_dict(untied.model) == before
        reloaded = TrainingState(tied_graph(), SGD())
        load_checkpoint!(reloaded, path)
        @test parameters(reloaded.model)["left.weights"] === parameters(reloaded.model)["right.weights"]
    end
end

@testset "Checkpoint round-trips SGD, RMSProp, and AdamW" begin
    x, y = Float32[1 2], Float32[2 4]
    mktempdir() do directory
        path = joinpath(directory, "optimizer.h5")
        for optimizer in (SGD(), SGD(; lr=0.02, momentum=0.75), RMSProp(; rho=0.95),
                          AdamW(; lr=0.002, weight_decay=0.08))
            state = TrainingState(dense_model(1, 1), optimizer)
            fit!(state, [(x, y)]; loss=QuadraticLoss(), epochs=2)
            save_checkpoint(path, state)
            resumed = TrainingState(dense_model(1, 1), Adam())
            load_checkpoint!(resumed, path)
            @test resumed.optimizer === optimizer
            @test resumed.slots[1].first == state.slots[1].first
            @test resumed.slots[1].second == state.slots[1].second
            for training in (state, resumed)
                fit!(training, [(x, y)]; loss=QuadraticLoss(), epochs=2)
            end
            @test state_dict(resumed.model) == state_dict(state.model)
        end
    end
end
