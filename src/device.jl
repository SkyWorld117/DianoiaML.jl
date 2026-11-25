device_of(::Union{Array,BitArray,AbstractRange}) = nothing
device_of(array::AbstractGPUArray) = device_backend(KA.get_backend(array))

device_backend(::KA.CPU) = nothing
device_backend(backend) = backend

wide_float(::Nothing) = Float64
wide_float(backend) = KA.supports_float64(backend) ? Float64 : Float32

gpu_precision_available(backend, ::Type) = true

function check_gpu_precision(backend, ::Type{T}) where {T}
    isbitstype(T) || throw(ArgumentError("GPU storage requires an isbits element type"))
    gpu_precision_available(backend, T) ||
        throw(ArgumentError("$(typeof(backend)) does not support $T on the selected GPU"))
end

function device_of(array::AbstractArray)
    root = parent(array)
    root === array || return device_of(root)
    backend = try
        KA.get_backend(array)
    catch error
        error isa ArgumentError || rethrow()
        return nothing
    end
    return device_backend(backend)
end

function device_of(arrays::Union{Tuple,NamedTuple})
    for array in arrays
        backend = device_of(array)
        backend === nothing || return backend
    end
    return nothing
end

gpu_array(array) = device_of(array) !== nothing
same_device(a, b) = device_of(a) === device_of(b)

function device_copy(backend, array::AbstractArray)
    check_gpu_precision(backend, eltype(array))
    output = KA.allocate(backend, eltype(array), size(array))
    copyto!(output, device_of(array) === backend ? array : Array(array))
    return output
end

function batch_on(backend, array::AbstractArray)
    return device_of(array) === backend ? array : device_copy(backend, array)
end

batch_on(::Nothing, array::Array) = array
batch_on(::Nothing, array::AbstractArray) = Array(array)
batch_on(backend, values::Union{Tuple,NamedTuple}) = map(x -> batch_on(backend, x), values)

zeros_like(array, ::Type{T}, dims::Tuple) where {T} = fill!(similar(array, T, dims), zero(T))
host_array(array::Array) = array
host_array(array::AbstractArray) = Array(array)
sum_map(f, array, init) = gpu_array(array) ? AK.mapreduce(f, +, array; init) : sum(f, array; init)

struct SquaredMagnitude{T} end
(::SquaredMagnitude{T})(x) where {T} = abs2(T(x))

function launch!(kernel, output, args...; ndrange=length(output), groupsize=256)
    isempty(output) || kernel(device_of(output), groupsize)(output, args...; ndrange)
    return output
end

"""Copy tensors or a model to a KernelAbstractions backend, preserving parameter ties."""
to_device(backend::KA.Backend, x::AbstractArray) = device_copy(backend, x)
to_device(::KA.CPU, x::AbstractArray) = Array(x)
to_device(backend::KA.Backend, xs::Union{Tuple,NamedTuple}) = map(x -> to_device(backend, x), xs)
