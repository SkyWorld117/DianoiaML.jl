abstract type AbstractLayerStack <: AbstractLayer end

mutable struct Hidden_Input_Layer
    output::Array{Float32}
end

mutable struct Hidden_Output_Layer
    δ::Array{Float32}
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

materialize_layer(constructor, shape; args...) = constructor(; input_shape=something(shape, ()), args...)

function materialize_layer(layer::AbstractLayer, shape; args...)
    isempty(args) || throw(ArgumentError("configure a layer before adding the instance to a model"))
    return layer
end

function prepare_layer(model, layer; args...)
    seen = IdDict{Any,Nothing}()
    validate_layer_tree!(seen, model)

    existing = layer_children(model)
    previous = isempty(existing) ? nothing : last(existing)
    previous_shape = previous === nothing ? nothing : previous.output_shape
    next = materialize_layer(layer, previous_shape; args...)
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

function initialize_layers!(layers, output_index, output_shape, batch)
    if length(layers) < output_index
        push!(layers, Hidden_Output_Layer(zeros(Float32, output_shape)))
    elseif size(layers[output_index].δ) != output_shape
        layers[output_index].δ = zeros(Float32, output_shape)
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
    return initialize_layers!(model.layers, model.num_layer + 2, (model.default_input_shape..., batch), batch)
end

function forward_layers!(layers, indices, input, training)
    for i in indices
        input = activate_layer(layers[i], input, training)
    end
    return input
end

function activate!(model::AbstractLayerStack, data::Array{Float32}; training::Bool=false)
    model.layers[1].output = data
    return forward_layers!(model.layers, 2:model.num_layer+1, data, training)
end

activate_layer(model::AbstractLayerStack, input, training) = activate!(model, input; training)

function update!(model::AbstractLayerStack, optimizer::String, input::Array{Float32}, delta,
                 α::Float64, parameters::Tuple, direction::Int=1)
    return backward_layers!(model.layers, 2:model.num_layer+1, input, delta, optimizer, α, parameters, direction)
end

function backward_layers!(layers, indices, first_input, delta, optimizer, α, parameters, direction)
    for i in Iterators.reverse(indices)
        input = i == first(indices) ? first_input : layers[i-1].output
        delta = update_layer!(layers[i], optimizer, input, delta, α, parameters, direction)
    end
    return delta
end
