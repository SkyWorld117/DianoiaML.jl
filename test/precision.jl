@testset "CPU precision" begin
    types = (Float16, BFloat16, Float32, Float64, BigFloat)
    labels = Dict(1 => 1, 2 => 2)

    for T in types
        @testset "$T" begin
            A = T in (Float16, BFloat16) ? Float32 : T
            model = Sequential(; precision=T)
            add_layer!(model, Dense; input_shape=(2,), layer_size=2,
                       activation_function=Softmax, randomization=false)
            layer = model.layers[2]
            state = TrainingState(model, Optimizer(:Adam; precision=T); seed=42)
            input = view(Float64[1 2 4; 2 4 8], :, 1:2)
            target = oneHot([1, 1], 2, labels; precision=T)
            prediction = forward!(state, input)
            @test eltype(prediction) === eltype(layer.weights) === eltype(target) === T
            @test prediction == fill(T(1//2), 2, 2)
            input_gradient = backward!(state, target; loss_function=Categorical_Cross_Entropy_Loss)
            @test eltype(input_gradient) === T
            @test iszero(input_gradient)
            derivative = gradients(state)["1.weights"]
            @test eltype(derivative) === A
            @test derivative == A[-3//4 -3//2; 3//4 3//2]
            step!(state; clip_norm=1)
            @test eltype(state.slots[1].first) === eltype(state.slots[1].second) === A
            @test all(isfinite, layer.weights)
            @test all(>(T(1//2)), forward!(state, input; training=false)[1, :])

            set_group!(state, layer; α=T(1//4))
            forward!(state, input)
            backward!(state, target; loss_function=Categorical_Cross_Entropy_Loss)
            mktempdir() do directory
                path = joinpath(directory, "model.h5")
                save_Sequential(model, path)
                loaded = load_Sequential(path)
                @test eltype(loaded.layers[2].weights) === T
                @test state_dict(loaded) == state_dict(model)
                resumed = TrainingState(loaded, Optimizer(:SGD))
                save_checkpoint(path, state)
                load_checkpoint!(resumed, path)
                @test gradients(resumed) == gradients(state)
                @test resumed.slots[1].first == state.slots[1].first
                @test typeof(resumed.slots[1].α) === typeof(state.slots[1].α)
                @test resumed.slots[1].α == state.slots[1].α
                step!(resumed)
                step!(state)
                @test state_dict(resumed.model) == state_dict(state.model)
            end

            dropout = Sequential(Dropout(input_shape=(8,), rate=1//2, precision=T))
            drop_state = TrainingState(dropout, Optimizer(:SGD); seed=21)
            mask_values = forward!(drop_state, ones(8, 8))
            @test eltype(mask_values) === T
            @test Set(mask_values) == Set(T[0, 2])
            @test backward!(drop_state, ones(8, 8)) == mask_values

            projection = Dense(input_shape=(1,), layer_size=1, activation_function=None,
                               randomization=false, precision=T)
            projection.weights .= 2
            graph = Graph(((1,),), Node(projection, 1), Node(share_parameters(projection), 1),
                          Node(Add((1,); precision=T), 2, 3); precision=T)
            graph_state = TrainingState(graph, Optimizer(:SGD))
            @test forward!(graph_state, ones(1, 2)) == fill(T(4), 1, 2)
            @test eltype(graph.output) === T
            @test backward!(graph_state, ones(1, 2)) == fill(T(4), 1, 2)
            @test gradients(graph_state)["1.weights"] == fill(A(2), 1, 1)
        end
    end
end

@testset "Spatial precision" begin
    x = reshape(Float64.(1:8), 2, 2, 1, 2)
    for Layer in (Conv2D, TConv2D)
        function spatial_model(T)
            model = Sequential(; precision=T)
            add_layer!(model, Layer; input_shape=(2, 2, 1), filter=1, kernel_size=(2, 2),
                       padding=1, strides=(2, 1), activation_function=None, randomization=false)
            model.layers[2].filters .= reshape([1, 2, 3, 4], 1, 1, 2, 2)
            return model
        end
        reference = TrainingState(spatial_model(Float64), Optimizer(:SGD))
        expected = copy(forward!(reference, x))
        delta = ones(size(expected))
        expected_input = copy(backward!(reference, delta))
        for T in (Float16, BFloat16, Float32, Float64, BigFloat)
            model = spatial_model(T)
            state = TrainingState(model, Optimizer(:SGD))
            prediction = forward!(state, x)
            @test eltype(prediction) === T
            @test prediction == expected
            @test backward!(state, delta) == expected_input
            @test gradients(state) == gradients(reference)
        end
    end

    for T in (Float16, BFloat16, Float64, BigFloat)
        model = Sequential(; precision=T)
        add_layer!(model, UpSampling2D; input_shape=(2, 2, 1), size=(2, 2), activation_function=None)
        add_layer!(model, MaxPooling2D; pool_size=(2, 2))
        add_layer!(model, Flatten)
        add_layer!(model, Constructive; shape=(2, 2, 1))
        state = TrainingState(model, Optimizer(:SGD))
        prediction = forward!(state, x)
        @test eltype(prediction) === T
        @test prediction == x
        derivative = backward!(state, ones(size(x)))
        @test eltype(derivative) === T
        @test derivative == ones(size(x))
    end
end

@testset "Nested precision and arbitrary precision round trips" begin
    @test eltype(dense_model(1, 1).layers[2].weights) === Float32
    setprecision(BigFloat, 384) do
        model = Sequential(; precision=BigFloat)
        branch = ResNet(; precision=BigFloat)
        add_layer!(branch, Dense; input_shape=(1,), layer_size=1, activation_function=None,
                   randomization=false)
        add_layer!(model, Residual(branch))
        layer = branch.layers[2]
        tiny = BigFloat(2)^(-320)
        layer.weights .= 1 + tiny
        layer.biases .= setprecision(BigFloat, 448) do
            BigFloat(2)^(-414)
        end
        state = TrainingState(model, Optimizer(:SGD; α=tiny, precision=BigFloat))
        @test only(forward!(state, ones(1, 1))) == 2 + tiny
        @test only(backward!(state, ones(1, 1))) == 2 + tiny
        saved = state_dict(model)
        mktempdir() do directory
            path = joinpath(directory, "big.h5")
            setprecision(BigFloat, 256) do
                save_Sequential(model, path)
                loaded = load_Sequential(path)
                @test state_dict(loaded) == saved
                @test precision(only(loaded.layers[2].resnet.layers[2].weights)) == 384
                @test precision(only(loaded.layers[2].resnet.layers[2].biases)) == 448
                save_checkpoint(path, state)
                load_checkpoint!(state, path)
                @test state_dict(model) == saved
                @test precision(state.optimizer.α) == 384
            end
        end
    end

    model = Sequential(; precision=Float64)
    add_layer!(model, Dense; precision=Float32, input_shape=(1,), layer_size=1,
               activation_function=None, randomization=false)
    add_layer!(model, Dense; layer_size=1, activation_function=None, randomization=false)
    @test eltype(model.layers[2].weights) === Float32
    @test eltype(model.layers[3].weights) === Float64
    state = TrainingState(model, Optimizer(:SGD))
    @test eltype(forward!(state, ones(Float16, 1, 2))) === Float64
    @test eltype(backward!(state, ones(1, 2))) === Float64
    mktempdir() do directory
        path = joinpath(directory, "mixed.h5")
        save_Sequential(model, path)
        loaded = load_Sequential(path)
        @test eltype(loaded.layers[2].weights) === Float32
        @test eltype(loaded.layers[3].weights) === Float64
    end
end

@testset "GAN precision" begin
    for T in (Float16, BFloat16, Float64, BigFloat)
        noise() = ones(1)
        model = GAN(noise; precision=T)
        model.add_Glayer(model, Dense; input_shape=(1,), layer_size=1,
                         activation_function=None, randomization=false)
        model.add_Dlayer(model, Dense; layer_size=1, activation_function=Sigmoid, randomization=false)
        model.layers[2].weights .= 1//2
        model.layers[3].weights .= 1//4
        input = fill(2.0, 1, 2)
        SGD.fit(; model, input_data=input, output_data=input, loss_function=Binary_Cross_Entropy_Loss,
                 monitor=Absolute, α=1//10, batch=2, epochs=1, shuffle=false, verbose=false)
        @test only(model.layers[2].weights) > T(1//2)
        @test eltype(model.activate(model)) === T
        mktempdir() do directory
            path = joinpath(directory, "gan.h5")
            save_GAN(model, path)
            loaded = load_GAN(path, noise)
            initialize!(loaded, 2)
            @test eltype(loaded.activate(loaded)) === T
            @test state_dict(loaded) == state_dict(model)
        end
    end
end
