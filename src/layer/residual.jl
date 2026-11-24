"""Add the output of a shape-preserving, single-input branch to its input."""
struct Residual{L<:AbstractLayer} <: AbstractLayer
    branch::L
    function Residual(branch::L) where {L<:AbstractLayer}
        layer = new{L}(branch)
        shape(layer)
        return layer
    end
end

layer_children(layer::Residual) = (; branch=layer.branch)
input_shape(layer::Residual) = input_shape(layer.branch)

function output_shape(layer::Residual, input)
    output = shape(layer.branch, input)
    check_shape(input, output)
    return input === nothing ? output : input
end

mutable struct ResidualCache
    output::Array
    gradient::Array
    ResidualCache() = new()
end

layer_cache(::Residual) = ResidualCache()

function sum_tensors!(cache, field, input, branch)
    size(input) == size(branch) || throw(DimensionMismatch("residual branch must preserve the input shape"))
    T = promote_type(eltype(input), eltype(branch))
    if !isdefined(cache, field) || size(getfield(cache, field)) != size(input) || eltype(getfield(cache, field)) !== T
        setfield!(cache, field, similar(input, T))
    end
    dest = getfield(cache, field)
    return dest .= input .+ branch
end

function forward!(::Residual, run::Execution, input::AbstractArray, mode)
    branch = execute!(run.children.branch, input, mode)
    return sum_tensors!(run.cache, :output, input, branch)
end

function backward!(::Residual, run::Execution, input, delta; compute_parameters)
    branch = differentiate!(run.children.branch, delta; compute_parameters)
    return sum_tensors!(run.cache, :gradient, delta, branch)
end
