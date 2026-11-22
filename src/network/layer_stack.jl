abstract type AbstractLayerStack <: AbstractLayer end

mutable struct Hidden_Input_Layer
    output::Union{Array{<:AbstractFloat},Tuple}
end

mutable struct Hidden_Output_Layer
    δ::Array{<:AbstractFloat}
end

function output_buffer!(output::Hidden_Output_Layer, prediction)
    if size(output.δ) != size(prediction) || eltype(output.δ) !== eltype(prediction)
        output.δ = similar(prediction)
    end
    return output.δ
end

@inline function Base.getproperty(model::AbstractLayerStack, name::Symbol)
    name === :add_layer && return add_layer!
    name === :input_shape && return model.num_layer == 0 ? () : model.layers[2].input_shape
    name === :output_shape && return model.num_layer == 0 ? () : last_layer(model).output_shape
    name === :output && return last_layer(model).output
    name === :δ && return model.layers[2].δ
    return invoke(Base.getproperty, Tuple{AbstractLayer,Symbol}, model, name)
end

Base.propertynames(model::AbstractLayerStack, private::Bool=false) =
    (invoke(Base.propertynames, Tuple{AbstractLayer,Bool}, model, private)...,
     :add_layer, :input_shape, :output_shape, :output, :δ)

last_layer(model::AbstractLayerStack) = model.layers[model.num_layer + 1]

layer_children(layer) = ()
layer_children(model::AbstractLayerStack) = @view model.layers[2:model.num_layer+1]

function validate_layer_tree!(seen, layer)
    haskey(seen, layer) &&
        throw(ArgumentError("a layer or network instance can appear only once in a model; use a fresh instance for each position"))
    seen[layer] = nothing
    for child in layer_children(layer)
        validate_layer_tree!(seen, child)
    end
    validate_connections(layer)
end

validate_layer_tree!(layer) = validate_layer_tree!(IdDict{Any,Nothing}(), layer)
validate_connections(layer) = nothing

function check_input_shape(layer, expected)
    if hasproperty(layer, :input_shape) && layer.input_shape != expected
        throw(DimensionMismatch("layer expects input shape $(layer.input_shape), received $expected"))
    end
end

function validate_stack_connections(layers)
    for i in 2:length(layers)
        previous, current = layers[i-1], layers[i]
        if previous isa AbstractLayerStack || current isa AbstractLayerStack
            check_input_shape(current, previous.output_shape)
        end
    end
end

validate_connections(model::AbstractLayerStack) = validate_stack_connections(layer_children(model))

accepts_precision(constructor) = false
function materialize_layer(constructor, shape, precision; args...)
    input_shape = something(shape, ())
    return accepts_precision(constructor) ? constructor(; input_shape, precision, args...) :
           constructor(; input_shape, args...)
end

function materialize_layer(layer::AbstractLayer, shape, precision; args...)
    isempty(args) || throw(ArgumentError("configure a layer before adding the instance to a model"))
    return layer
end

function prepare_layer(model, layer; args...)
    seen = IdDict{Any,Nothing}()
    validate_layer_tree!(seen, model)

    existing = layer_children(model)
    previous = isempty(existing) ? nothing : last(existing)
    previous_shape = previous === nothing ? nothing : previous.output_shape
    next = materialize_layer(layer, previous_shape, model_precision(model); args...)
    next isa AbstractLayerStack && next.num_layer == 0 &&
        throw(ArgumentError("a nested network requires at least one layer"))
    validate_layer_tree!(seen, next)
    if previous !== nothing && (layer isa AbstractLayer || next isa AbstractLayerStack || previous isa AbstractLayerStack)
        check_input_shape(next, previous_shape)
    end
    return next
end

"""
    add_layer!(model, layer; kwargs...)

Append a configured layer or network instance, or construct a layer from a
constructor and keyword arguments. A nested network retains its own parameters.
"""
function add_layer!(model::AbstractLayerStack, layer; args...)
    next = prepare_layer(model, layer; args...)
    insert!(model.layers, model.num_layer + 2, next)
    model.num_layer += 1
    model.default_input_shape = next.output_shape
end

function initialize_layers!(layers, output_index, output_shape, batch, precision)
    T = something(precision_type(layers[output_index-1]), precision)
    if length(layers) < output_index
        push!(layers, Hidden_Output_Layer(zeros(T, output_shape)))
    elseif size(layers[output_index].δ) != output_shape || eltype(layers[output_index].δ) !== T
        layers[output_index].δ = zeros(T, output_shape)
    end
    for i in 2:output_index-1
        initialize!(layers[i], batch)
    end
end

function initialize!(model::AbstractLayerStack, batch::Int)
    model.num_layer > 0 || throw(ArgumentError("$(nameof(typeof(model))) requires at least one layer"))
    batch > 0 || throw(ArgumentError("batch size must be positive"))
    validate_layer_tree!(model)
    model.default_input_shape = model.output_shape
    return initialize_layers!(model.layers, model.num_layer + 2, (model.default_input_shape..., batch),
                              batch, model_precision(model))
end

function forward_layers!(layers, indices, input, training)
    for i in indices
        input = activate_layer(layers[i], input, training)
    end
    return input
end

function activate_layer(model::AbstractLayerStack, data, training)
    data = layer_input(model, data)
    model.layers[1].output = data
    return forward_layers!(model.layers, 2:model.num_layer+1, data, training)
end

activate!(model::AbstractLayerStack, data::Union{AbstractArray{<:AbstractFloat},Tuple}; training::Bool=false) =
    activate_layer(model, data, training)

supports_backward(model::AbstractLayerStack) = all(supports_backward, layer_children(model))

function backward!(model::AbstractLayerStack, input, delta)
    return propagate_layers!(model.layers, 2:model.num_layer+1, input, delta)
end

function propagate_layers!(layers, indices, input, delta)
    for i in Iterators.reverse(indices)
        previous = i == first(indices) ? input : layers[i-1].output
        delta = propagate_layer!(layers[i], previous, delta)
    end
    return delta
end

function update!(model::AbstractLayerStack, optimizer::String, input::Array{<:AbstractFloat}, delta,
                 α::Real, parameters::Tuple, direction::Int=1)
    return backward_layers!(model.layers, 2:model.num_layer+1, layer_input(model, input),
                            delta, optimizer, α, parameters, direction)
end

function backward_layers!(layers, indices, first_input, delta, optimizer, α, parameters, direction)
    if all(i -> supports_backward(layers[i]), indices)
        result = propagate_layers!(layers, indices, first_input, delta)
        direction == 0 || legacy_step!(layers, indices, optimizer, α, parameters, direction)
        return result
    end
    for i in Iterators.reverse(indices)
        input = i == first(indices) ? first_input : layers[i-1].output
        delta = update_layer!(layers[i], optimizer, input, delta, α, parameters, direction)
    end
    return delta
end
