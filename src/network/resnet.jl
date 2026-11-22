module resnet
    using ..DianoiaML: AbstractLayer, AbstractLayerStack, Hidden_Input_Layer, Hidden_Output_Layer, add_layer!
    import ..DianoiaML: update!, precision_type
    using ..DianoiaML: checked_precision, model_precision

    mutable struct ResNet{T<:AbstractFloat} <: AbstractLayerStack
        layers::Vector{Any}
        num_layer::Int
        loss::AbstractFloat
        default_input_shape::Tuple

        ResNet(; precision::Type{T}=Float32) where {T<:AbstractFloat} =
            new{T}(Any[Hidden_Input_Layer(checked_precision(T)[])], 0, 0.0, ())
    end

    precision_type(::ResNet{T}) where {T} = T

    function ResNet(first::AbstractLayer, rest::AbstractLayer...; precision=model_precision(first))
        model = ResNet(; precision)
        add_layer!(model, first)
        for layer in rest
            add_layer!(model, layer)
        end
        return model
    end

    function update!(model::ResNet, optimizer::String, α::Real, parameters...; direction::Int=1)
        update!(model, optimizer, model.layers[1].output, model.layers[end].δ, α, parameters, direction)
        return nothing
    end
end
