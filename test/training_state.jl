@testset "Separate backward, optimizer groups, and accumulation" begin
    layer = configured_dense(ones(Float32, 1, 1))
    state = TrainingState(Sequential(layer), Optimizer(:SGD; α=0.1))
    x = Float32[1 2 3]
    y = zero(x)
    forward!(state, x[:, 1:1])
    @test backward!(state, y[:, 1:1]; loss_function=Quadratic_Loss) == Float32[2;;]
    forward!(state, x[:, 2:3])
    backward!(state, y[:, 2:3]; loss_function=Quadratic_Loss)
    @test layer.weights == ones(Float32, 1, 1)
    @test layer.biases == zeros(Float32, 1)
    @test only(gradients(state)["1.weights"]) ≈ 28f0 / 3
    @test only(gradients(state)["1.biases"]) ≈ 4f0

    whole = configured_dense(ones(Float32, 1, 1))
    full = TrainingState(Sequential(whole), Optimizer(:SGD; α=0.1))
    forward!(full, x)
    backward!(full, y; loss_function=Quadratic_Loss)
    step!(state; clip_norm=2)
    step!(full; clip_norm=2)
    @test layer.weights ≈ whole.weights
    @test layer.biases ≈ whole.biases
    @test hypot(only(layer.weights) - 1, only(layer.biases)) ≈ 0.2

    set_group!(state, layer.biases; trainable=false)
    set_group!(state, layer.weights; α=0.0)
    before = state_dict(state.model)
    forward!(state, x)
    backward!(state, y; loss_function=Quadratic_Loss)
    step!(state)
    @test state_dict(state.model) == before
end

function tied_graph()
    weight = configured_dense(fill(2f0, 1, 1))
    tied = share_parameters(weight)
    return Graph(((1,), (1,)), Node(weight, 1), Node(tied, 2), Node(Add((1,)), 3, 4))
end

@testset "Graph branches share parameters and sum fan-out cotangents" begin
    graph = tied_graph()
    state = TrainingState(graph, Optimizer(:SGD; α=0.2))
    params = parameters(graph)
    @test params["1.weights"] === params["2.weights"]
    @test length(gradients(state)) == 2
    set_group!(state, params["1.biases"]; trainable=false)
    a, b = Float32[1 2], Float32[3 4]
    @test forward!(state, (a, b)) == Float32[8 12]
    @test graph.nodes[1].layer.output !== graph.nodes[2].layer.output
    @test backward!(state, ones(Float32, 1, 2)) == (fill(2f0, 1, 2), fill(2f0, 1, 2))
    @test only(gradients(state)["1.weights"]) == 5f0
    step!(state)
    @test only(params["1.weights"]) ≈ 1f0
    @test only(params["2.biases"]) == 0f0

    inner = Graph(((1,),), Node(configured_dense(fill(2f0, 1, 1)), 1),
                  Node(configured_dense(fill(3f0, 1, 1)), 1); outputs=(2, 3))
    outer = Graph(((1,),), Node(inner, 1), Node(Select(2, ((1,), (1,))), 2),
                  Node(Add((1,)), 1, 3); outputs=(4, 2))
    state = TrainingState(outer, Optimizer(:SGD))
    x = Float32[1 2]
    @test forward!(state, x) == (4f0 .* x, (2f0 .* x, 3f0 .* x))
    @test backward!(state, (one.(x), (one.(x), nothing))) == fill(6f0, 1, 2)
    forward!(state, x)
    @test backward!(state, nothing; loss_function=(prediction, _) ->
                    (one.(prediction[1]), nothing)) == fill(4f0, 1, 2)
    @test_throws ArgumentError Graph(((1,),), Node(configured_dense(ones(Float32, 1, 1)), 2))
end

@testset "Training state detects replaced parameters" begin
    layer = configured_dense(ones(Float32, 1, 1))
    state = TrainingState(Sequential(layer), Optimizer(:SGD))
    layer.weights = copy(layer.weights)
    @test_throws ArgumentError forward!(state, ones(Float32, 1, 1))
end

mutable struct MaskLayer <: AbstractLayer
    input_shape::Tuple
    output_shape::Tuple
    output::Array{Float32,3}
end

DianoiaML.initialize!(::MaskLayer, ::Int) = nothing
DianoiaML.supports_backward(::MaskLayer) = true
function DianoiaML.activate!(layer::MaskLayer, input::Tuple)
    features, mask = input
    layer.output = features .* mask
end
DianoiaML.backward!(::MaskLayer, input::Tuple, delta) = (delta .* input[2], nothing)

@testset "Iterable batches, masks, and variable token lengths" begin
    projection = Dense(input_shape=(1, 1), layer_size=1, activation_function=None, randomization=false)
    mask = MaskLayer(((1, 1), (1, 1)), (1, 1), zeros(Float32, 0, 0, 0))
    graph = Graph(((1, 1), (1, 1)), Node(projection, 1), Node(mask, 3, 2))
    state = TrainingState(graph, Optimizer(:SGD; α=0.05))
    x = reshape(Float64.(1:6), 1, 3, 2)
    batch(tokens) = ((view(x, :, 1:tokens, :), trues(1, tokens, 2)), Float32.(view(x, :, 1:tokens, :)))
    fit!(state, () -> (batch(tokens) for tokens in (2, 3)); loss_function=Quadratic_Loss,
         epochs=2, accumulate=2, verbose=false)
    @test state.epochs == 2
    @test state.slots[1].step == 2
    @test size(forward!(state, first(batch(2)); training=false)) == (1, 2, 2)
    @test size(forward!(state, first(batch(3)); training=false)) == (1, 3, 2)
end

@testset "Checkpoint resumes optimizers, partial batches, and dropout" begin
    checkpoint_model() = Sequential(Dropout(input_shape=(2,), rate=0.5),
                                    configured_dense(Float32[0.2 -0.3]))
    state = TrainingState(checkpoint_model(), Optimizer(:Adam; α=0.02); seed=19)
    set_group!(state, parameters(state.model)["2.biases"]; α=0.005)
    x = Float32[1 2 3; 2 1 0]
    y = Float32[0.2 0.3 0.4]
    fit!(state, [(x, y)]; loss_function=Quadratic_Loss, verbose=false)
    forward!(state, x[:, 1:1])
    backward!(state, y[:, 1:1]; loss_function=Quadratic_Loss)

    mktempdir() do folder
        path = joinpath(folder, "training.h5")
        save_checkpoint(path, state)
        save_checkpoint(path, state)
        resumed = TrainingState(checkpoint_model(), Optimizer(:SGD); seed=0)
        load_checkpoint!(resumed, path)
        @test state_dict(resumed.model) == state_dict(state.model)
        @test gradients(resumed) == gradients(state)
        @test resumed.samples == resumed.microbatches == resumed.epochs == 1
        @test resumed.optimizer.kind == "Adam"
        @test resumed.slots[2].α == 0.005

        @test forward!(resumed, x[:, 2:3]) == forward!(state, x[:, 2:3])
        for training in (resumed, state)
            backward!(training, y[:, 2:3]; loss_function=Quadratic_Loss)
            step!(training)
        end
        @test state_dict(resumed.model) == state_dict(state.model)
        @test resumed.slots[1].first == state.slots[1].first
        @test resumed.slots[1].step == state.slots[1].step == 2

        tied = TrainingState(tied_graph(), Optimizer(:Adam))
        save_checkpoint(path, tied)
        untied_graph = Graph(((1,), (1,)), Node(configured_dense(ones(Float32, 1, 1)), 1),
                             Node(configured_dense(ones(Float32, 1, 1)), 2), Node(Add((1,)), 3, 4))
        untied = TrainingState(untied_graph, Optimizer(:Adam))
        before = state_dict(untied_graph)
        @test_throws ArgumentError load_checkpoint!(untied, path)
        @test state_dict(untied_graph) == before
        reloaded_tie = TrainingState(tied_graph(), Optimizer(:SGD))
        load_checkpoint!(reloaded_tie, path)
        @test parameters(reloaded_tie.model)["1.weights"] === parameters(reloaded_tie.model)["2.weights"]
    end
end
