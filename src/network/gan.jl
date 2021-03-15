module gan
    using .Threads

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

        default_input_size::Int64
        default_input_filter::Int64
        default_input2D_size::Tuple

        G_range::UnitRange{Int64}
        D_range::UnitRange{Int64}

        function GAN(noise_generator)
            new(Any[Hidden_Input_Layer(Float32[])], init_GAN, activate_GAN, activate_Generator, activate_Discriminator, update_GAN, noise_generator, GAN_add_Glayer, GAN_add_Dlayer, 0, 0, 0.0, 0, 1, ())
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        propagation_units::Array{Float32}
    end

    function GAN_add_Glayer(model::GAN, layer::Any;args...)
        try
            push!(model.layers, layer(;input_size=model.default_input_size, input_filter=model.default_input_filter, input2D_size=model.default_input2D_size, args...))
        catch e
            push!(model.layers, layer(;input_size=model.default_input_size, args...))
        end
        model.default_input_size = model.layers[end].layer_size
        try
            kwargs = Dict(args)
            model.default_input_filter = kwargs[:filter]
            input2D_size = model.layers[end].input2D_size
            #padding = kwargs[:padding]
            kernel_size = kwargs[:kernel_size]
            model.default_input2D_size = (input2D_size[1]-kernel_size[1]+1, input2D_size[2]-kernel_size[2]+1)
        catch e
        end
        model.num_Glayer += 1
    end
    function GAN_add_Dlayer(model::GAN, layer::Any;args...)
        try
            push!(model.layers, layer(;input_size=model.default_input_size, input_filter=model.default_input_filter, input2D_size=model.default_input2D_size, args...))
        catch e
            push!(model.layers, layer(;input_size=model.default_input_size, args...))
        end
        model.default_input_size = model.layers[end].layer_size
        try
            kwargs = Dict(args)
            model.default_input_filter = kwargs[:filter]
            input2D_size = model.layers[end].input2D_size
            #padding = kwargs[:padding]
            kernel_size = kwargs[:kernel_size]
            model.default_input2D_size = (input2D_size[1]-kernel_size[1]+1, input2D_size[2]-kernel_size[2]+1)
        catch e
        end
        model.num_Dlayer += 1
    end

    function init_GAN(model::GAN, mini_batch::Int64)
        if length(model.layers) == model.num_Glayer+model.num_Dlayer+1
            push!(model.layers, Hidden_Output_Layer(Float32[]))
        end
        @threads for i in 2:length(model.layers)-1
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
        model.loss += monitor.func(model.layers[end-1].output, current_output_data)

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
