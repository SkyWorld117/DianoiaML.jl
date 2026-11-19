module gan
    using ..DianoiaML: activate_layer
    import ..DianoiaML

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end

    mutable struct Hidden_Output_Layer
        δ::Array{Float32}
    end

    mutable struct GAN
        layers::Vector{Any}
        initialize::Any
        activate::Any
        activate_Generator::Any
        activate_Discriminator::Any
        update::Any
        noise_generator::Any

        add_Glayer::Any
        add_Dlayer::Any
        num_Glayer::Int
        num_Dlayer::Int
        loss::Float64
        default_input_shape::Tuple
        G_range::UnitRange{Int}
        D_range::UnitRange{Int}
        real_target::Matrix{Float32}
        fake_target::Matrix{Float32}

        function GAN(noise_generator)
            new(Any[Hidden_Input_Layer(Float32[])], init_GAN, activate_GAN, activate_Generator,
                activate_Discriminator, update_GAN, noise_generator, GAN_add_Glayer, GAN_add_Dlayer,
                0, 0, 0.0, (), 2:1, 2:1)
        end
    end

    function GAN_add_Glayer(model::GAN, layer; args...)
        model.num_Dlayer == 0 || throw(ArgumentError("add generator layers before discriminator layers"))
        next = layer(; input_shape=model.default_input_shape, args...)
        push!(model.layers, next)
        model.num_Glayer += 1
        model.G_range = 2:model.num_Glayer+1
        model.default_input_shape = next.output_shape
    end

    function GAN_add_Dlayer(model::GAN, layer; args...)
        next = layer(; input_shape=model.default_input_shape, args...)
        insert!(model.layers, model.num_Glayer + model.num_Dlayer + 2, next)
        model.num_Dlayer += 1
        model.D_range = model.num_Glayer+2:model.num_Glayer+model.num_Dlayer+1
        model.default_input_shape = next.output_shape
    end

    function init_GAN(model::GAN, mini_batch::Int)
        model.num_Glayer > 0 && model.num_Dlayer > 0 || throw(ArgumentError("GAN requires a generator and discriminator"))
        mini_batch > 0 || throw(ArgumentError("batch size must be positive"))
        input_shape = (model.layers[2].input_shape..., mini_batch)
        if size(model.layers[1].output) != input_shape
            model.layers[1].output = zeros(Float32, input_shape)
        end

        output_shape = (model.default_input_shape..., mini_batch)
        output_index = last(model.D_range) + 1
        if length(model.layers) < output_index
            push!(model.layers, Hidden_Output_Layer(zeros(Float32, output_shape)))
        elseif size(model.layers[end].δ) != output_shape
            model.layers[end].δ = zeros(Float32, output_shape)
        end
        for i in 2:output_index-1
            layer = model.layers[i]
            layer.initialize(layer, mini_batch)
        end
    end

    function init_targets!(model::GAN, batch::Int)
        model.default_input_shape in ((1,), (2,)) ||
            throw(DimensionMismatch("GAN training requires one or two discriminator outputs per example"))
        outputs = only(model.default_input_shape)
        if !isdefined(model, :real_target) || size(model.real_target) != (outputs, batch)
            model.real_target = zeros(Float32, outputs, batch)
            model.real_target[end, :] .= 1f0
            model.fake_target = zeros(Float32, outputs, batch)
            outputs == 2 && (model.fake_target[1, :] .= 1f0)
        end
    end

    function forward(model::GAN, layers, input, training)
        for i in layers
            layer = model.layers[i]
            activate_layer(layer, input, training)
            input = layer.output
        end
        return input
    end

    function activate_Generator(model::GAN; training::Bool=false)
        input = model.layers[1].output
        features = prod(model.layers[2].input_shape)
        for first in 1:features:length(input)
            sample = model.noise_generator()
            length(sample) == features || throw(DimensionMismatch("noise generator must return one sample"))
            copyto!(input, first, sample, 1, features)
        end
        return forward(model, model.G_range, input, training)
    end

    function activate_Discriminator(model::GAN, data::Array{Float32}; training::Bool=false)
        return forward(model, model.D_range, data, training)
    end

    function activate_GAN(model::GAN; training::Bool=false)
        generated = model.activate_Generator(model; training)
        return model.activate_Discriminator(model, generated; training)
    end

    function backward!(model::GAN, layers, first_input, delta, optimizer, α, parameters, direction)
        for i in Iterators.reverse(layers)
            layer = model.layers[i]
            input = i == first(layers) ? first_input : model.layers[i-1].output
            if direction == 1
                layer.update(layer, optimizer, input, delta, α, parameters)
            else
                layer.update(layer, optimizer, input, delta, α, parameters, direction)
            end
            delta = layer.δ
        end
        return delta
    end

    discriminator_step(::Tuple{}, offset) = ()
    discriminator_step(parameters, offset) = (2 * parameters[1] + offset, Base.tail(parameters)...)

    function update_GAN(model::GAN, current_input_data::Array{Float32}, current_output_data::Array{Float32},
                        loss_function, monitor, optimizer::String, α::Float64, parameters...)
        init_targets!(model, size(current_input_data, ndims(current_input_data)))
        last_layer = model.layers[last(model.D_range)]
        output_delta = model.layers[end].δ

        real = model.activate_Discriminator(model, current_input_data; training=true)
        model.loss += monitor.func(real, model.real_target)
        delta = DianoiaML.output_gradient!(output_delta, last_layer, model.real_target, loss_function)
        backward!(model, model.D_range, current_input_data, delta, optimizer, α, discriminator_step(parameters, -1), 1)

        generated = model.activate_Generator(model; training=true)
        fake = model.activate_Discriminator(model, generated; training=true)
        model.loss += monitor.func(fake, model.fake_target)
        delta = DianoiaML.output_gradient!(output_delta, last_layer, model.fake_target, loss_function)
        backward!(model, model.D_range, generated, delta, optimizer, α, discriminator_step(parameters, 0), 1)

        model.activate_Discriminator(model, generated; training=true)
        delta = DianoiaML.output_gradient!(output_delta, last_layer, model.real_target, loss_function)
        generator_delta = backward!(model, model.D_range, generated, delta, optimizer, α, parameters, 0)
        backward!(model, model.G_range, model.layers[1].output, generator_delta, optimizer, α, parameters, 1)
        return nothing
    end
end
