module DianoiaMLOpenCLExt

using DianoiaML
using KernelAbstractions: supports_float64, synchronize
using OpenCL: cl, OpenCLBackend, OpenCLKernels
using PrecompileTools: @setup_workload, @compile_workload
using Random: Xoshiro

DianoiaML.gpu_precision_available(backend::OpenCLBackend, ::Type{Float16}) =
    "cl_khr_fp16" in OpenCLKernels.backend_device(backend).extensions
DianoiaML.gpu_precision_available(backend::OpenCLBackend, ::Type{Float64}) =
    supports_float64(backend)
# OpenCL's SPIR-V compiler does not enable the bfloat type.
DianoiaML.gpu_precision_available(::OpenCLBackend, ::Type{BFloat16}) = false

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
    devices = [device for platform in cl.platforms() for device in cl.available_devices(platform, :gpu)]
    if !isempty(devices)
        device = first(devices)
        cl.device!(device)
        backend = OpenCLBackend()
        @compile_workload begin
            for T in (Float32, Float16, BFloat16, Float64)
                DianoiaML.gpu_precision_available(backend, T) || continue
                convolution_workload(backend, T)
            end
            transformer_workload(backend)
            synchronize(backend)
        end
    end
end

end
