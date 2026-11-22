abstract type AbstractLayerStack end

mutable struct Hidden_Input_Layer
    output::Array{Float32}
end

mutable struct Hidden_Output_Layer
    δ::Array{Float32}
end

@inline function Base.getproperty(model::AbstractLayerStack, name::Symbol)
    name === :add_layer && return add_layer!
    name === :initialize && return initialize!
    name === :activate && return activate!
    name === :update && return update!
    return getfield(model, name)
end

Base.propertynames(model::AbstractLayerStack, private::Bool=false) =
    (fieldnames(typeof(model))..., :add_layer, :initialize, :activate, :update)

function add_layer!(model::AbstractLayerStack, layer; args...)
    next = layer(; input_shape=model.default_input_shape, args...)
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

function backward_layers!(layers, indices, first_input, delta, optimizer, α, parameters, direction)
    for i in Iterators.reverse(indices)
        input = i == first(indices) ? first_input : layers[i-1].output
        delta = update_layer!(layers[i], optimizer, input, delta, α, parameters, direction)
    end
    return delta
end
