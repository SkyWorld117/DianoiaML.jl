module gan
    mutable struct GAN
        layers::Array{Any}
        initialize::Any
        activate::Any
        activate_Generator::Any
        activate_Discriminator::Any
        update::Any

        noise_generator::Any

        add_Glayer::Any
        add_Dlayer::Any
        num_Glayer::Int64
        num_Dlayer::Int64

        loss::Float64

        default_input_shape::Tuple

        G_range::UnitRange{Int64}
        D_range::UnitRange{Int64}

        function GAN(noise_generator)
            new(Any[Hidden_Input_Layer(Float32[])], init_GAN, activate_GAN, activate_Generator, activate_Discriminator, update_GAN, noise_generator, GAN_add_Glayer, GAN_add_Dlayer, 0, 0, 0.0, ())
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        δ::Array{Float32}
    end

    function GAN_add_Glayer(model::GAN, layer::Any;args...)
        push!(model.layers, layer(;input_shape=model.default_input_shape, args...))
        model.num_Glayer += 1
        model.default_input_shape = model.layers[end].output_shape
    end
    function GAN_add_Dlayer(model::GAN, layer::Any;args...)
        push!(model.layers, layer(;input_shape=model.default_input_shape, args...))
        model.num_Dlayer += 1
        model.default_input_shape = model.layers[end].output_shape
    end

    function init_GAN(model::GAN, mini_batch::Int64)
        if length(model.layers) == model.num_Glayer+model.num_Dlayer+1
            push!(model.layers, Hidden_Output_Layer(zeros(Float32, model.layers[end].output_shape..., mini_batch)))
        end
        for i in 2:length(model.layers)-1
            model.layers[i].initialize(model.layers[i], mini_batch)
        end
        model.G_range = 2:model.num_Glayer+1
        model.D_range = model.num_Glayer+2:model.num_Glayer+model.num_Dlayer+1
    end

    function activate_Generator(model::GAN)
        model.layers[1].output = model.noise_generator()
        for i in model.G_range
            model.layers[i].activate(model.layers[i], model.layers[i-1].output)
        end
        return model.layers[model.num_Glayer+1].output
    end
    function activate_Discriminator(model::GAN, data::Array{Float32})
        model.layers[1+model.num_Glayer].output = data
        for i in model.D_range
            model.layers[i].activate(model.layers[i], model.layers[i-1].output)
        end
    end
    function activate_GAN(model::GAN)
        model.layers[1].output = model.noise_generator()
        for i in 2:length(model.layers)-1
            model.layers[i].activate(model.layers[i], model.layers[i-1].output)
        end
    end

    function update_GAN(model::GAN, current_input_data::Array{Float32}, current_output_data::Array{Float32}, loss_function::Any, monitor::Any, optimizer::String, α::Float64, parameters...)
        model.activate_Discriminator(model, current_input_data)
        loss_function.prop!(model.layers[end].δ, model.layers[end-1].output, reshape([0.0f0; 1.0f0], (2,1)))
        for i in Iterators.reverse(model.D_range)
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].δ, α, parameters)
        end
        model.loss += monitor.func(model.layers[end-1].output, current_output_data)

        model.activate(model)
        loss_function.prop!(model.layers[end].δ, model.layers[end-1].output, reshape([0.0f0; 1.0f0], (2,1)))
        for i in Iterators.reverse(model.D_range)
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].δ, α, parameters, -1)
        end
        for i in Iterators.reverse(model.G_range)
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].δ, α, parameters)
        end
    end
end
