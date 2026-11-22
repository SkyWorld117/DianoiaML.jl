abstract type AbstractLayer end

function backward! end
parameter_gradients(layer) = (;)
persistent_buffers(layer) = (;)
supports_backward(layer) = false

struct TrainingMode{R}
    rng::R
end

# Retain the property-based protocol for custom layers and models.
function call_operation(method, object, property, args...; kwargs...)
    operation = getproperty(object, property)
    operation === method && throw(MethodError(method, (object, args...)))
    return operation(object, args...; kwargs...)
end

initialize!(object, batch) = call_operation(initialize!, object, :initialize, batch)
activate!(object, args...; kwargs...) = call_operation(activate!, object, :activate, args...; kwargs...)
update!(object, args...) = call_operation(update!, object, :update, args...)
save_layer(layer, file, id) = call_operation(save_layer, layer, :save_layer, file, id)
load_layer!(layer, file, id) = call_operation(load_layer!, layer, :load_layer, file, id)
load_layer!(::AbstractLayer, file, id::String) = nothing

# Built-in layers expose the same calls through methods instead of instance fields.
@inline function Base.getproperty(layer::AbstractLayer, name::Symbol)
    name === :initialize && return initialize!
    name === :activate && return activate!
    name === :update && return update!
    name === :save_layer && return save_layer
    name === :load_layer && return load_layer!
    return getfield(layer, name)
end

Base.propertynames(layer::AbstractLayer, private::Bool=false) =
    (fieldnames(typeof(layer))..., :initialize, :activate, :update, :save_layer, :load_layer)

function activate_layer(layer, input, training)
    activate!(layer, layer_input(layer, input))
    return layer.output
end

function propagate_layer!(layer, input, delta)
    derivative = backward!(layer, layer_input(layer, input), cotangent_as(layer.output, delta))
    return cotangent_as(input, derivative)
end

function update_layer!(layer, optimizer, input, delta, α, parameters, direction)
    input = layer_input(layer, input)
    delta = cotangent_as(layer.output, delta)
    if direction == 1
        update!(layer, optimizer, input, delta, α, parameters)
    else
        update!(layer, optimizer, input, delta, α, parameters, direction)
    end
    return layer.δ
end
