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

zeros_like(array, ::Type{T}, dims::Tuple) where {T} = fill_tensor!(similar(array, T, dims), zero(T))
host_array(array::Array) = array
host_array(array::AbstractArray) = Array(array)
sum_map(f, array, init) = gpu_array(array) ? AK.mapreduce(f, +, array; init) : sum(f, array; init)

function reduce_sum!(f, dest, input; backend=device_of(dest))
    backend === nothing && return sum!(f, dest, input)
    return AK.mapreducedim!(f, +, dest, input; backend=recording_backend(backend), init=zero(eltype(dest)))
end

reduce_sum!(::Type{T}, dest, input; backend=device_of(dest)) where {T} =
    reduce_sum!(ConvertTo{T}(), dest, input; backend)

struct SquaredMagnitude{T} end
(::SquaredMagnitude{T})(x) where {T} = abs2(T(x))

function launch!(kernel, output, args...; ndrange=length(output), groupsize=256)
    if !isempty(output)
        target = recording_backend(device_of(output))
        kernel(target, groupsize)(output, args...; ndrange)
    end
    return output
end

const KI = KA.KernelInterface

struct RecordedLaunch{K,A}
    kernel::K
    groups::Dims{3}
    items::Dims{3}
    args::A
end

const kernel_recording = Base.ScopedValues.ScopedValue{Union{Nothing,Vector{RecordedLaunch}}}(nothing)

struct RecordingBackend{B<:KA.Backend} <: KA.Backend
    backend::B
    launches::Vector{RecordedLaunch}
end

function recording_backend(backend)
    recording = kernel_recording[]
    return recording === nothing ? backend : RecordingBackend(backend, recording)
end

KI.argconvert(b::RecordingBackend, arg) = KI.argconvert(b.backend, arg)
KI.max_work_group_size(b::RecordingBackend) = KI.max_work_group_size(b.backend)
KI.max_work_group_dims(b::RecordingBackend) = KI.max_work_group_dims(b.backend)
KI.max_num_groups(b::RecordingBackend) = KI.max_num_groups(b.backend)
KI.supports_linear_subgroups(b::RecordingBackend) = KI.supports_linear_subgroups(b.backend)
KI.supports_shuffle(b::RecordingBackend, T::Type) = KI.supports_shuffle(b.backend, T)
KI.sub_group_size(b::RecordingBackend) = KI.sub_group_size(b.backend)
KI.multiprocessor_count(b::RecordingBackend) = KI.multiprocessor_count(b.backend)
KI.allocate(b::RecordingBackend, T::Type, dims::Tuple; kwargs...) = KI.allocate(b.backend, T, dims; kwargs...)
KI.max_work_group_size(kernel::KI.Kernel{<:RecordingBackend}) = KI.max_work_group_size(kernel.kern)
AK.reduce_tuning(b::RecordingBackend, T::Type) = AK.reduce_tuning(b.backend, T)
AK.fill_on!(dest, value, ::RecordingBackend) = fill_tensor!(dest, value)

function KI.kernel_function(b::RecordingBackend, f, tt=Tuple{}; kwargs...)
    return KI.Kernel(b, KI.kernel_function(b.backend, f, tt; kwargs...))
end

function KI.launch(kernel::KI.Kernel{<:RecordingBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple)
    push!(kernel.backend.launches, RecordedLaunch(kernel.kern, groups, items, args))
    return
end

function record_launches(f)
    launches = RecordedLaunch[]
    Base.ScopedValues.@with kernel_recording => launches f()
    return launches
end

# Metal indirect buffers can dispatch compute kernels, but cannot record the
# runtime's blits or broadcasts. Use the same recorder for those tensor writes.
recording_writes(dest) = kernel_recording[] !== nothing && gpu_array(dest)

function broadcast_tensor!(f, dest, args...)
    bc = Base.Broadcast.broadcasted(f, args...)
    recording_writes(dest) || return Base.Broadcast.materialize!(dest, bc)
    bc = Base.Broadcast.instantiate(Base.Broadcast.Broadcasted(bc.f, bc.args, axes(dest)))
    bc = Base.Broadcast.preprocess(dest, bc)
    return launch!(broadcast_tensor_gpu!, dest, bc)
end

function map_tensor(f, input, ::Type{T}) where {T}
    recording_writes(input) || return f.(input)
    return broadcast_tensor!(f, similar(input, T), input)
end

function copy_tensor!(dest, src)
    recording_writes(dest) || return copyto!(dest, src)
    return broadcast_tensor!(identity, dest, src)
end

function fill_tensor!(dest, value)
    recording_writes(dest) || return fill!(dest, value)
    return broadcast_tensor!(identity, dest, value)
end

"""Copy tensors or a model to a KernelAbstractions backend, preserving parameter ties."""
to_device(backend::KA.Backend, x::AbstractArray) = device_copy(backend, x)
to_device(::KA.CPU, x::AbstractArray) = Array(x)
to_device(backend::KA.Backend, xs::Union{Tuple,NamedTuple}) = map(x -> to_device(backend, x), xs)
