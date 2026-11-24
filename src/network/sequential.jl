"""Compose layers in order. Optional input_shape describes one example."""
struct Sequential{L<:Tuple,S} <: AbstractLayer
    layers::L
    input::S
    function Sequential(layers::AbstractLayer...; input_shape=nothing)
        isempty(layers) && throw(ArgumentError("Sequential needs at least one layer"))
        model = new{typeof(layers),typeof(input_shape)}(layers, input_shape)
        shape(model)
        return model
    end
end

layer_children(model::Sequential) = model.layers
input_shape(model::Sequential) = model.input === nothing ? input_shape(first(model.layers)) : model.input

function output_shape(model::Sequential, input)
    for layer in model.layers
        input = shape(layer, input)
    end
    return input
end

function forward!(::Sequential, run::Execution, input, mode)
    for child in run.children
        input = execute!(child, input, mode)
    end
    return input
end

function backward!(::Sequential, run::Execution, input, delta; compute_parameters)
    for child in Iterators.reverse(run.children)
        delta = differentiate!(child, delta; compute_parameters)
    end
    return delta
end
