using DianoiaML
using KernelAbstractions: CPU, synchronize
using Random: Xoshiro, seed!
using Test

const GPU_NAME = isempty(ARGS) ? :cuda : Symbol(first(ARGS))
const GPU_RUNTIME = if GPU_NAME === :cuda
    import CUDACore
    CUDACore
elseif GPU_NAME === :amd
    import AMDGPU
    AMDGPU
elseif GPU_NAME === :oneapi
    import oneAPI
    oneAPI
elseif GPU_NAME === :metal
    import Metal
    Metal
else
    error("Choose cuda, amd, oneapi or metal as the first argument")
end

const GPU_BACKEND = gpu_backend(GPU_NAME)
GPU_RUNTIME.allowscalar(false)
println("GPU tests on ", repr(MIME"text/plain"(), GPU_RUNTIME.device()))

gpu(x) = to_device(GPU_BACKEND, x)
on_test_gpu(x) = DianoiaML.device_of(x) === GPU_BACKEND
host(x::AbstractArray) = Array(x)
host(::Nothing) = nothing
host(xs::Union{Tuple,NamedTuple}) = map(host, xs)
random_tensor(dims...) = randn(Xoshiro(76), Float32, dims...)

function check_tensor(actual::AbstractArray, expected::AbstractArray; atol=3f-5, rtol=5f-4)
    @test on_test_gpu(actual)
    @test Array(actual) ≈ expected atol=atol rtol=rtol
end

check_tensor(actual::Nothing, expected::Nothing; kwargs...) = @test actual === expected

function check_tensor(actual::Union{Tuple,NamedTuple}, expected::Union{Tuple,NamedTuple}; kwargs...)
    @test keys(actual) == keys(expected)
    for (value, reference) in zip(actual, expected)
        check_tensor(value, reference; kwargs...)
    end
end

function check_layer(model, inputs; atol=3f-5, rtol=5f-4)
    on_device = gpu(model)
    cpu_state, gpu_state = TrainingState(model, SGD()), TrainingState(on_device, SGD())
    expected = copy(forward!(cpu_state, inputs))
    actual = forward!(gpu_state, gpu(inputs))
    check_tensor(actual, expected; atol, rtol)
    inference_inputs = isempty(parameters(model)) ? gpu(inputs) : inputs
    check_tensor(predict(on_device, inference_inputs), expected; atol, rtol)
    upstream = random_tensor(size(expected)...)
    check_tensor(backward!(gpu_state, gpu(upstream)), backward!(cpu_state, upstream); atol, rtol)
    expected_gradients = gradients(cpu_state)
    for (name, gradient) in gradients(gpu_state)
        check_tensor(gradient, expected_gradients[name]; atol, rtol)
    end
    return on_device
end

if "--smoke" in ARGS
    @testset "DianoiaML GPU smoke ($GPU_NAME)" begin
        seed!(123)
        model = Sequential(Conv2D(1, 2; kernel_size=(2, 2), activation=ReLU()),
                           Flatten(), Dense(18, 2); input_shape=(4, 4, 1))
        x = random_tensor(4, 4, 1, 3)
        y = Float32[1 0 1; 0 1 0]
        on_device = check_layer(model, x)
        for (candidate, inputs, labels) in ((model, x, y), (on_device, gpu(x), gpu(y)))
            state = TrainingState(candidate, AdamW(); seed=1)
            fit!(state, batches(inputs, labels; batch_size=2, shuffle=false);
                 loss=CrossEntropyWithLogits(), monitor=Classification(), clip_norm=1, verbose=false)
        end
        check_tensor(predict(on_device, gpu(x)), predict(model, x))

        tokens = random_tensor(4, 3, 2)
        mask = Bool[1 1; 1 0; 0 0]
        check_layer(TransformerBlock(4; heads=2, causal=true), (tokens, mask))
        synchronize(GPU_BACKEND)
    end
else
    include("gpu_suite.jl")
end
