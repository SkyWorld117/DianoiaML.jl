module DianoiaMLCUDACoreExt

using CUDACore: CUDACore, CUDABackend, synchronize
using DianoiaML
using PrecompileTools: @setup_workload, @compile_workload
using Random: Xoshiro

DianoiaML.gpu_backend(::Val{:cuda}) = CUDABackend()

include("replay_cuda.jl")

function convolution_workload(backend, ::Type{T}) where {T}
    model = to_device(backend, Sequential(
        Conv2D(1, 2; kernel_size=(3, 3), activation=ReLU(), precision=T, randomization=false),
        MaxPooling2D((2, 2)),
        Conv2D(2, 4; kernel_size=(3, 3), activation=ReLU(), precision=T, randomization=false),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, 8; activation=ReLU(), precision=T, randomization=false),
        Dense(8, 3; precision=T, randomization=false);
        input_shape=(28, 28, 1),
    ))
    # Four images exercise the split reduction for convolution weight gradients.
    # The fifth also exercises the final partial batch.
    inputs = to_device(backend, fill(T(1//2), 28, 28, 1, 5))
    labels = zeros(T, 3, 5)
    labels[1, :] .= one(T)
    targets = to_device(backend, labels)

    state = TrainingState(model, Adam(); seed=1)
    fit!(state, batches(inputs, targets; batch_size=4, rng=Xoshiro(1));
         loss=CrossEntropyWithLogits(), monitor=Classification(), verbose=false)
    predict(model, inputs)
    return
end

function transformer_workload(backend)
    model = to_device(backend, Sequential(
        TransformerBlock(8; heads=2, causal=true, dropout=1//8),
        TransformerBlock(8; heads=2, causal=true, dropout=1//8),
        Dense(8, 3),
    ))
    tokens = to_device(backend, fill(0.5f0, 8, 4, 2))
    mask = to_device(backend, trues(4, 2))
    targets = to_device(backend, zeros(Float32, 3, 4, 2))
    state = TrainingState(model, AdamW(); seed=1)
    fit!(state, batches((tokens, mask), targets; batch_size=2, shuffle=false);
         loss=MeanSquaredError(), clip_norm=1, verbose=false)
    predict(model, tokens)
    return
end

@setup_workload begin
    # CUDACore 6.4 can persist relocatable device binaries in package images.
    if pkgversion(CUDACore) >= v"6.4" && CUDACore.functional() && CUDACore.ndevices() > 0
        backend = CUDABackend()
        @compile_workload begin
            for T in (Float32, Float16, BFloat16, Float64)
                convolution_workload(backend, T)
            end
            transformer_workload(backend)
            synchronize()
        end
    end
end

end
