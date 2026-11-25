"""Concatenate graph inputs along one axis within each example."""
struct Concatenate <: AbstractLayer
    dims::Int
    function Concatenate(dims::Int)
        dims > 0 || throw(ArgumentError("Concatenate requires a positive axis"))
        return new(dims)
    end
end

layer_cache(::Concatenate) = ActivationCache()

function output_shape(layer::Concatenate, inputs)
    inputs === nothing && return nothing
    known = filter(!isnothing, inputs)
    isempty(known) && return nothing
    rank = length(first(known))
    layer.dims <= rank && all(s -> length(s) == rank, known) ||
        throw(DimensionMismatch("Concatenate requires equal ranks and an axis within each example"))
    return ntuple(rank) do dim
        extents = filter(x -> x isa Int, map(s -> s[dim], known))
        if dim == layer.dims
            return length(extents) == length(inputs) ? sum(extents) : nothing
        end
        isempty(extents) && return nothing
        all(==(first(extents)), extents) || throw(DimensionMismatch("Concatenate input shapes differ"))
        return first(extents)
    end
end

function concatenation_ranges(layer, inputs)
    widths = map(input -> size(input, layer.dims), inputs)
    ends = accumulate(+, widths)
    return map((stop, width) -> stop-width+1:stop, ends, widths)
end

function forward!(layer::Concatenate, cache::ActivationCache, inputs::Tuple, mode)
    length(inputs) >= 2 || throw(ArgumentError("Concatenate requires at least two inputs"))
    batch = batch_examples(inputs)
    dims = (output_shape(layer, map(x -> Base.front(size(x)), inputs))..., batch)
    T = promote_type(map(eltype, inputs)...)
    source = first(inputs)
    if !isdefined(cache, :output) || size(cache.output) != dims ||
       eltype(cache.output) !== T || !same_device(cache.output, source)
        cache.output = similar(source, T, dims)
    end
    for (input, indices) in zip(inputs, concatenation_ranges(layer, inputs))
        selectdim(cache.output, layer.dims, indices) .= input
    end
    return cache.output
end

function backward!(layer::Concatenate, ::ActivationCache, inputs, ∇y; compute_parameters=true)
    return map(r -> selectdim(∇y, layer.dims, r), concatenation_ranges(layer, inputs))
end

"""Subtract the second input tensor from the first."""
struct Subtract <: AbstractLayer end
layer_cache(::Subtract) = ActivationCache()
output_shape(::Subtract, inputs) = output_shape(Add(), inputs)

function forward!(::Subtract, cache::ActivationCache, inputs::Tuple{AbstractArray,AbstractArray}, mode)
    return sum_tensors!(cache, :output, inputs..., -)
end

backward!(::Subtract, ::ActivationCache, inputs, ∇y; compute_parameters=true) = (∇y, .-∇y)

"""Permute the axes of each example while keeping the batch axis last."""
struct Permute{N} <: AbstractLayer
    order::NTuple{N,Int}
    inverse::NTuple{N,Int}
    function Permute(order::NTuple{N,Int}) where {N}
        sort(collect(order)) == collect(1:N) || throw(ArgumentError("invalid axis permutation"))
        inverse = ntuple(i -> findfirst(==(i), order)::Int, N)
        return new{N}(order, inverse)
    end
end

layer_cache(::Permute) = ActivationCache()
input_shape(::Permute{N}) where {N} = ntuple(_ -> nothing, N)
output_shape(layer::Permute, input) =
    input === nothing ? input_shape(layer) : map(i -> input[i], layer.order)

function permute_examples!(cache, field, input, order)
    ndims(input) == length(order) + 1 || throw(DimensionMismatch("Permute input rank differs"))
    axes_order = (order..., ndims(input))
    dims = map(i -> size(input, i), axes_order)
    if !isdefined(cache, field) || size(getfield(cache, field)) != dims ||
       eltype(getfield(cache, field)) !== eltype(input) || !same_device(getfield(cache, field), input)
        setfield!(cache, field, similar(input, dims))
    end
    return permutedims!(getfield(cache, field), input, axes_order)
end

forward!(layer::Permute, cache::ActivationCache, input, mode) =
    permute_examples!(cache, :output, input, layer.order)

backward!(layer::Permute, cache::ActivationCache, input, ∇y; compute_parameters=true) =
    permute_examples!(cache, :gradient, ∇y, layer.inverse)
