module sequential
    using ..DianoiaML: AbstractLayerStack, Hidden_Input_Layer, Hidden_Output_Layer, backward_layers!, activate!
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

    function update!(model::Sequential, input::Array{Float32}, target::Array{Float32},
                     loss_function, monitor, optimizer::String, α::Float64, parameters...)
        prediction = activate!(model, input; training=true)
        delta = DianoiaML.output_gradient!(model.layers[end].δ, model.layers[end-1], target, loss_function)
        backward_layers!(model.layers, 2:model.num_layer+1, input, delta, optimizer, α, parameters, 1)
        model.loss += monitor.func(prediction, target)
    end
end
