@testset "Layers for built-in networks" begin
    @testset "Batch normalization" begin
        layer = gpu(BatchNorm2D(2; momentum=1))
        state = TrainingState(layer, SGD())
        x = reshape(Float32[1, 3, 2, 4, 5, 7, 6, 8], 2, 1, 2, 2)
        ∇y = reshape(Float32[1, -2, 3, 2, -1, 2, 4, -3], size(x))
        μ = sum(x; dims=(1, 2, 4)) ./ 4
        z = x .- μ
        σ² = sum(abs2, z; dims=(1, 2, 4)) ./ 4
        inverse_std = inv.(sqrt.(σ² .+ layer.epsilon))
        normalized = z .* inverse_std
        check_tensor(forward!(state, gpu(x)), normalized)
        @test Array(layer.running_mean) ≈ vec(μ)
        @test Array(layer.running_variance) ≈ vec(σ²) .* (4f0 / 3)
        ∑∇y = sum(∇y; dims=(1, 2, 4))
        ∑∇yz = sum(∇y .* normalized; dims=(1, 2, 4))
        expected = inverse_std .* (∇y .- (∑∇y .+ normalized .* ∑∇yz) ./ 4)
        check_tensor(backward!(state, gpu(∇y)), expected)
        @test Array(gradients(state)["bias"]) ≈ vec(∑∇y) ./ 2
        @test Array(gradients(state)["scale"]) ≈ vec(∑∇yz) ./ 2
        snapshot = state_dict(layer)
        check_tensor(predict(layer, x), z ./ sqrt.(σ² .* (4f0 / 3) .+ layer.epsilon))
        copy_layer = gpu(BatchNorm2D(2))
        load_state!(copy_layer, snapshot)
        @test state_dict(copy_layer) == snapshot
        @test Array(predict(copy_layer, x)) == Array(predict(layer, x))
    end

    check_layer(GlobalMeanPool2D(), random_tensor(3, 5, 2, 3))
    check_layer(Permute((3, 1, 2)), random_tensor(2, 3, 4, 2))
    check_layer(Concatenate(3), (random_tensor(2, 3, 4, 2), random_tensor(2, 3, 5, 2)))
    check_layer(Subtract(), (random_tensor(4, 2), random_tensor(4, 2)))

    @testset "Embedding gradients and indexed labels" begin
        embedding = gpu(Embedding(4, 3))
        state = TrainingState(embedding, SGD())
        ids = Int32[2 1; 2 3]
        weights = Array(embedding.weights)
        check_tensor(forward!(state, gpu(ids)), reshape(weights[:, vec(ids)], 3, 2, 2))
        @test backward!(state, gpu(ones(Float32, 3, 2, 2))) === nothing
        @test Array(gradients(state)["weights"]) ≈ repeat(Float32[0.5 1 0.5 0], 3, 1)
        @test_throws ArgumentError predict(embedding, Int32[1 5])

        positions = check_layer(PositionalEmbedding(3, 5), random_tensor(3, 2, 2))
        state = TrainingState(positions, SGD())
        forward!(state, gpu(ones(Float32, 3, 2, 2)))
        backward!(state, gpu(ones(Float32, 3, 2, 2)))
        @test Array(gradients(state)["weights"]) == hcat(ones(Float32, 3, 2), zeros(Float32, 3, 3))
        @test_throws DimensionMismatch predict(positions, ones(Float32, 3, 6, 2))

        logits = random_tensor(3, 2, 2) .+ 1000f0
        labels = Int32[1 3; 2 1]
        one_hot = zeros(Float32, size(logits))
        for j in axes(labels, 2), i in axes(labels, 1)
            one_hot[labels[i, j], i, j] = 1
        end
        indexed, dense = IndexedCrossEntropyWithLogits(), CrossEntropyWithLogits()
        device_logits, device_labels = gpu(logits), gpu(labels)
        @test indexed(device_logits, device_labels) ≈ dense(logits, one_hot) / 2
        reference = loss_gradient!(similar(logits), dense, logits, one_hot) ./ 2
        check_tensor(loss_gradient!(similar(device_logits), indexed, device_logits, device_labels), reference)
        @test_throws ArgumentError indexed(device_logits, fill(Int32(4), size(labels)))
        @test_throws DimensionMismatch indexed(device_logits, Int32[1, 2])
    end
end

@testset "Built-in networks train on the GPU" begin
    cases = (
        (ResNet18(3, 3; width=4), random_tensor(32, 32, 3, 3), Int32[1, 2, 3], (3, 3),
         IndexedCrossEntropyWithLogits()),
        (UNet(2, 3; channels=(4, 8, 16)), random_tensor(16, 24, 2, 3),
         rand(Xoshiro(8), Int32(1):Int32(3), 16, 24, 3), (3, 16, 24, 3), IndexedCrossEntropyWithLogits()),
        (GPT(12; width=16, heads=2, depth=2, context_length=8),
         rand(Xoshiro(8), Int32(1):Int32(12), 8, 3), rand(Xoshiro(9), Int32(1):Int32(12), 8, 3),
         (12, 8, 3), IndexedCrossEntropyWithLogits()),
        (NBEATS(12, 3; width=16, basis=4, blocks=2), random_tensor(12, 3), random_tensor(3, 3), (3, 3),
         MeanSquaredError()),
    )
    for (model, x, y, expected_shape, loss) in cases
        @test shape(model, Base.front(size(x))) == Base.front(expected_shape)
        on_device, inputs, targets = gpu(model), gpu(x), gpu(y)
        state = TrainingState(on_device, AdamW())
        before = Dict(name => Array(p) for (name, p) in parameters(on_device))
        fit!(state, batches(inputs, targets; batch_size=2, shuffle=false); loss, verbose=false)
        out = predict(on_device, inputs)
        @test state.epochs == 1
        @test on_test_gpu(out)
        @test size(out) == expected_shape
        @test isfinite(loss(out, targets))
        @test any(Array(value) != before[name] for (name, value) in parameters(on_device))
    end

    @test_throws DimensionMismatch shape(UNet(2, 3; channels=(4, 8, 16)), (15, 24, 2))
    @test shape(NBEATS(12, 3; width=8, basis=4, blocks=1)) == (3,)

    model = gpu(GPT(12; width=16, heads=2, depth=2, context_length=8))
    tokens = rand(Xoshiro(29), Int32(1):Int32(12), 8, 2)
    a = predict(model, tokens)
    prefix = predict(model, tokens[1:5, :])
    check_tensor(prefix, Array(a)[:, 1:5, :])

    if DianoiaML.gpu_precision_available(GPU_BACKEND, Float16)
        model = gpu(GPT(12; width=16, heads=2, depth=1, context_length=8, precision=Float16))
        state = TrainingState(model, AdamW())
        fit!(state, [(gpu(tokens[1:7, :]), gpu(tokens[2:8, :]))];
             loss=IndexedCrossEntropyWithLogits(), verbose=false)
        output = predict(model, tokens)
        @test eltype(output) === Float16
        @test all(isfinite, Array(output))
    end
end
