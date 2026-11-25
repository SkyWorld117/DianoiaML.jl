device_of(::Union{Array,BitArray,AbstractRange}) = nothing

"""Return a KernelAbstractions backend by name after its runtime package is loaded."""
function gpu_backend(name::Symbol)
    backend = gpu_backend(Val(name))
    (KA.functional(backend) === false || iszero(KA.ndevices(backend))) &&
        throw(ArgumentError("GPU backend :$name has no working device"))
    return backend
end

gpu_backend(::Val{name}) where {name} =
    throw(ArgumentError("GPU backend :$name is unavailable. Load the corresponding runtime package first."))

wide_float(::Nothing) = Float64
wide_float(backend) = KA.supports_float64(backend) ? Float64 : Float32

gpu_precision_available(backend, ::Type) = true
gpu_precision_available(backend, ::Type{Float64}) = KA.supports_float64(backend)

function check_gpu_precision(backend, ::Type{T}) where {T}
    isbitstype(T) || throw(ArgumentError("GPU storage requires an isbits element type"))
    gpu_precision_available(backend, T) ||
        throw(ArgumentError("$(typeof(backend)) does not support $T on the selected device"))
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
    return backend isa KA.CPU ? nothing : backend
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

matching_buffer(::Nothing, value) = false
matching_buffer(buffer::AbstractArray, value::AbstractArray) =
    size(buffer) == size(value) && eltype(buffer) === eltype(value) && same_device(buffer, value)

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

function reduce_sum!(f, dest, input; backend=device_of(dest))
    backend === nothing && return sum!(f, dest, input)
    return AK.mapreducedim!(f, +, dest, input; backend, init=zero(eltype(dest)))
end

reduce_sum!(::Type{T}, dest, input; backend=device_of(dest)) where {T} =
    reduce_sum!(ConvertTo{T}(), dest, input; backend)

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
