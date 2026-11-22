module sequential
    using ..DianoiaML: AbstractLayer, AbstractLayerStack, Hidden_Input_Layer, Hidden_Output_Layer,
                       add_layer!, last_layer, activate!
    import ..DianoiaML
    import ..DianoiaML: update!

    mutable struct Sequential <: AbstractLayerStack
        # The public layer list also accepts custom layers implementing the property protocol.
        layers::Vector{Any}
        num_layer::Int
        loss::Float64
        default_input_shape::Tuple

        Sequential() = new(Any[Hidden_Input_Layer(Float32[])], 0, 0.0, ())
    end

    """
        Sequential(layers...)

    Build a model from configured layers and nonempty `Sequential` or `ResNet`
    blocks. Use a distinct instance at each position in the model.
    """
    function Sequential(first::AbstractLayer, rest::AbstractLayer...)
        model = Sequential()
        add_layer!(model, first)
        for layer in rest
            add_layer!(model, layer)
        end
        return model
    end

    function update!(model::Sequential, input::Array{Float32}, target::Array{Float32},
                     loss_function, monitor, optimizer::String, α::Float64, parameters...)
        prediction = activate!(model, input; training=true)
        if size(model.layers[end].δ) != size(prediction)
            model.layers[end].δ = similar(prediction)
        end
        delta = DianoiaML.output_gradient!(model.layers[end].δ, last_layer(model), target, loss_function)
        update!(model, optimizer, input, delta, α, parameters)
        model.loss += monitor.func(prediction, target)
    end
end
