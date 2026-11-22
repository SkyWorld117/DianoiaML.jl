module resnet
    using ..DianoiaML: AbstractLayer, AbstractLayerStack, Hidden_Input_Layer, Hidden_Output_Layer, add_layer!
    import ..DianoiaML: update!

    mutable struct ResNet <: AbstractLayerStack
        layers::Vector{Any}
        num_layer::Int
        loss::Float64
        default_input_shape::Tuple

        ResNet() = new(Any[Hidden_Input_Layer(Float32[])], 0, 0.0, ())
    end

    function ResNet(first::AbstractLayer, rest::AbstractLayer...)
        model = ResNet()
        add_layer!(model, first)
        for layer in rest
            add_layer!(model, layer)
        end
        return model
    end

    function update!(model::ResNet, optimizer::String, α::Float64, parameters...; direction::Int=1)
        update!(model, optimizer, model.layers[1].output, model.layers[end].δ, α, parameters, direction)
        return nothing
    end
end
