module gan
    Gdef = 0
    Ddef = 0

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
        temp_Glayers::Array{Any}
        temp_Dlayers::Array{Any}

        loss::Float64

        G_range::UnitRange{Int64}
        D_range::UnitRange{Int64}

        function GAN(noise_generator)
            new(Any[Hidden_Input_Layer(Float32[])], init_GAN, activate_GAN, activate_Generator, activate_Discriminator, update_GAN, noise_generator, GAN_add_Glayer, GAN_add_Dlayer, 0, 0, Any[], Any[], 0.0)
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        propagation_units::Array{Float32}
    end

    function GAN_add_Glayer(model::GAN, layer::Any)
        global Gdef
        push!(model.temp_Glayers, layer)
        model.num_Glayer += 1
        Gdef = layer.layer_size
    end
    function GAN_add_Dlayer(model::GAN, layer::Any)
        global Ddef
        push!(model.temp_Dlayers, layer)
        model.num_Dlayer += 1
        Ddef = layer.layer_size
    end

    function init_GAN(model::GAN, mini_batch::Int64)
        if length(model.layers) < model.num_Glayer+model.num_Dlayer+1
            model.layers = vcat(model.layers, model.temp_Glayers, model.temp_Dlayers)
        end
        if length(model.layers) == model.num_Glayer+model.num_Dlayer+1
            push!(model.layers, Hidden_Output_Layer(Float32[]))
        end
        Threads.@threads for i in 2:length(model.layers)-1
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
        model.layers[end].propagation_units = loss_function.prop(model.layers[end-1].output, reshape([0.0f0; 1.0f0], (2,1)))
        for i in Iterators.reverse(model.D_range)
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].propagation_units, α, parameters)
        end

        model.activate(model)
        model.layers[end].propagation_units = loss_function.prop(model.layers[end-1].output, reshape([0.0f0; 1.0f0], (2,1)))
        for i in Iterators.reverse(model.D_range)
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].propagation_units, α, parameters, -1)
        end
        for i in Iterators.reverse(model.G_range)
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].propagation_units, α, parameters)
        end
    end
end
