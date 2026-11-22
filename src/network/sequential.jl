module sequential
    using ..DianoiaML: AbstractLayer, AbstractLayerStack, Hidden_Input_Layer, Hidden_Output_Layer,
                       add_layer!, last_layer, activate!, output_buffer!
    import ..DianoiaML
    import ..DianoiaML: update!, precision_type
    using ..DianoiaML: checked_precision, model_precision

    mutable struct Sequential{T<:AbstractFloat} <: AbstractLayerStack
        # The public layer list also accepts custom layers implementing the property protocol.
        layers::Vector{Any}
        num_layer::Int
        loss::AbstractFloat
        default_input_shape::Tuple

        Sequential(; precision::Type{T}=Float32) where {T<:AbstractFloat} =
            new{T}(Any[Hidden_Input_Layer(checked_precision(T)[])], 0, 0.0, ())
    end

    """
        Sequential(layers...)

    Build a model from configured layers and nonempty `Sequential` or `ResNet`
    blocks. Use a distinct instance at each position in the model.
    """
    function Sequential(first::AbstractLayer, rest::AbstractLayer...; precision=model_precision(first))
        model = Sequential(; precision)
        add_layer!(model, first)
        for layer in rest
            add_layer!(model, layer)
        end
        return model
    end

    precision_type(::Sequential{T}) where {T} = T

    function update!(model::Sequential, input::Array{<:AbstractFloat}, target::Array{<:AbstractFloat},
                     loss_function, monitor, optimizer::String, α::Real, parameters...)
        prediction = activate!(model, input; training=true)
        output_delta = output_buffer!(model.layers[end], prediction)
        delta = DianoiaML.output_gradient!(output_delta, last_layer(model), target, loss_function)
        update!(model, optimizer, model.layers[1].output, delta, α, parameters)
        model.loss += monitor.func(prediction, target)
    end
end
