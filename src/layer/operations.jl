abstract type AbstractLayer end

function forward! end
function backward! end

layer_parameters(layer) = (;)
layer_gradients(layer, cache) = (;)
persistent_buffers(layer) = (;)
layer_children(layer) = ()
layer_cache(layer) = nothing

# Shape hints contain known axes. A missing hint or a nothing extent is
# unconstrained. The batch axis is checked separately when data arrives.
input_shape(layer) = nothing
output_shape(layer, input) = nothing

function check_shape(expected::Tuple, actual::Tuple)
    if !any(x -> x isa Colon, expected) && !any(x -> x isa Colon, actual) && length(expected) != length(actual)
        throw(DimensionMismatch("expected shape $expected, received $actual"))
    end
    for (wanted, supplied) in zip(expected, actual)
        (wanted isa Colon || supplied isa Colon) && break
        wanted === nothing && continue
        supplied === nothing && continue
        if wanted isa Tuple && supplied isa Tuple
            check_shape(wanted, supplied)
        elseif wanted != supplied
            throw(DimensionMismatch("expected shape $expected, received $actual"))
        end
    end
    return actual
end

check_shape(expected, actual) = actual

function shape(layer::AbstractLayer, input=input_shape(layer))
    check_shape(input_shape(layer), input)
    return output_shape(layer, input)
end

function feature_shape(input, width)
    return input === nothing || isempty(input) ? (width, Colon()) : (width, Base.tail(input)...)
end

const Tensor = Union{AbstractArray,Tuple,NamedTuple}

struct TrainingMode{R}
    rng::R
end

mutable struct Execution{L,C,K}
    layer::L
    cache::C
    children::K
    input::Union{Nothing,Tensor}
    data::Union{Nothing,Tensor}
    output::Union{Nothing,Tensor}
end

function execution(layer)
    children = map(execution, layer_children(layer))
    return Execution(layer, layer_cache(layer), children, nothing, nothing, nothing)
end

forward!(layer, run::Execution, input, mode) = forward!(layer, run.cache, input, mode)
backward!(layer, run::Execution, input, delta; compute_parameters) =
    backward!(layer, run.cache, input, delta; compute_parameters)

function execute!(run::Execution, input, mode=false)
    run.input = input
    data = run.data = float_input(precision_type(run.layer), input)
    return run.output = forward!(run.layer, run, data, mode)
end

function differentiate!(run::Execution, delta; compute_parameters=true)
    upstream = cotangent_as(run.output, delta)
    derivative = backward!(run.layer, run, run.data, upstream; compute_parameters)
    return cotangent_as(run.input, derivative)
end

cpu_batch(input::Array) = input
cpu_batch(input::AbstractArray) = Array(input)
cpu_batch(inputs::Union{Tuple,NamedTuple}) = map(cpu_batch, inputs)

"""Predict using inference buffers independent of any pending training pass."""
predict(model::AbstractLayer, input) = execute!(execution(model), cpu_batch(input))

function walk_layers(f, layer, prefix="")
    f(layer, prefix)
    for (name, child) in pairs(layer_children(layer))
        walk_layers(f, child, prefix * string(name) * ".")
    end
end

function walk_executions(f, run::Execution, prefix="")
    f(run, prefix)
    for (name, child) in pairs(run.children)
        walk_executions(f, child, prefix * string(name) * ".")
    end
end

"""Return named references to a model's parameters, including shared arrays."""
function parameters(model)
    result = Dict{String,Array{<:AbstractFloat}}()
    walk_layers(model) do layer, prefix
        for (name, value) in pairs(layer_parameters(layer))
            result[prefix * string(name)] = value
        end
    end
    return result
end
