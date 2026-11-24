struct Reshape{D} <: AbstractLayer
    dimensions::D
    function Reshape(dimensions::Tuple{Vararg{Int}})
        all(>(0), dimensions) || throw(ArgumentError("reshape extents must be positive"))
        new{typeof(dimensions)}(dimensions)
    end
    Reshape() = new{Nothing}(nothing)
end

"""Collapse all feature axes while keeping the examples on the last axis."""
Flatten() = Reshape()

function output_shape(layer::Reshape, input)
    count = input === nothing || any(d -> !(d isa Int), input) ? nothing : prod(input)
    dims = layer.dimensions
    dims === nothing && return (count,)
    count === nothing || count == prod(dims) || throw(DimensionMismatch("reshape changes the feature count"))
    return dims
end

function forward!(layer::Reshape, ::Nothing, input, mode)
    ndims(input) >= 2 || throw(DimensionMismatch("reshape needs feature and example axes"))
    return reshape(input, shape(layer, Base.front(size(input)))..., size(input, ndims(input)))
end

backward!(::Reshape, ::Nothing, input, delta; compute_parameters=true) = reshape(delta, size(input))
