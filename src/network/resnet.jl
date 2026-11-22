module resnet
    using ..DianoiaML: AbstractLayerStack, Hidden_Input_Layer, Hidden_Output_Layer, backward_layers!
    import ..DianoiaML: update!

    mutable struct ResNet <: AbstractLayerStack
        layers::Vector{Any}
        num_layer::Int
        loss::Float64
        default_input_shape::Tuple

        ResNet() = new(Any[Hidden_Input_Layer(Float32[])], 0, 0.0, ())
    end

    function update!(model::ResNet, optimizer::String, α::Float64, parameters...; direction::Int=1)
        backward_layers!(model.layers, 2:model.num_layer+1, model.layers[1].output,
                         model.layers[end].δ, optimizer, α, parameters, direction)
        return nothing
    end
end
