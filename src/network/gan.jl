module gan
    using ..DianoiaML: Hidden_Input_Layer, initialize_layers!, forward_layers!, backward_layers!
    using ..DianoiaML: prepare_layer, validate_layer_tree!, validate_stack_connections
    import ..DianoiaML
    using ..DianoiaML: checked_precision, output_buffer!
    import ..DianoiaML: precision_type, initialize!, activate!, update!, layer_children, validate_connections

    mutable struct GAN{F,T<:AbstractFloat}
        layers::Vector{Any}
        noise_generator::F
        num_Glayer::Int
        num_Dlayer::Int
        loss::AbstractFloat
        default_input_shape::Tuple
        G_range::UnitRange{Int}
        D_range::UnitRange{Int}
        real_target::Matrix{T}
        fake_target::Matrix{T}

        function GAN(noise_generator::F; precision::Type{T}=Float32) where {F,T<:AbstractFloat}
            new{F,T}(Any[Hidden_Input_Layer(checked_precision(T)[])], noise_generator, 0, 0, 0.0, (), 2:1, 2:1)
        end
    end

    precision_type(::GAN{F,T}) where {F,T} = T

    @inline function Base.getproperty(model::GAN, name::Symbol)
        name === :initialize && return initialize!
        name === :activate && return activate!
        name === :update && return update!
        name === :activate_Generator && return activate_generator!
        name === :activate_Discriminator && return activate_discriminator!
        name === :add_Glayer && return add_generator_layer!
        name === :add_Dlayer && return add_discriminator_layer!
        return getfield(model, name)
    end

    Base.propertynames(model::GAN, private::Bool=false) =
        (fieldnames(typeof(model))..., :initialize, :activate, :update,
         :activate_Generator, :activate_Discriminator, :add_Glayer, :add_Dlayer)

    layer_children(model::GAN) = @view model.layers[2:model.num_Glayer+model.num_Dlayer+1]
    validate_connections(model::GAN) = validate_stack_connections(layer_children(model))

    function add_generator_layer!(model::GAN, layer; args...)
        model.num_Dlayer == 0 || throw(ArgumentError("add generator layers before discriminator layers"))
        next = prepare_layer(model, layer; args...)
        push!(model.layers, next)
        model.num_Glayer += 1
        model.G_range = 2:model.num_Glayer+1
        model.default_input_shape = next.output_shape
    end

    function add_discriminator_layer!(model::GAN, layer; args...)
        next = prepare_layer(model, layer; args...)
        insert!(model.layers, model.num_Glayer + model.num_Dlayer + 2, next)
        model.num_Dlayer += 1
        model.D_range = model.num_Glayer+2:model.num_Glayer+model.num_Dlayer+1
        model.default_input_shape = next.output_shape
    end

    function initialize!(model::GAN{F,T}, mini_batch::Int) where {F,T}
        model.num_Glayer > 0 && model.num_Dlayer > 0 || throw(ArgumentError("GAN requires a generator and discriminator"))
        mini_batch > 0 || throw(ArgumentError("batch size must be positive"))
        validate_layer_tree!(model)
        input_shape = (model.layers[2].input_shape..., mini_batch)
        if size(model.layers[1].output) != input_shape
            model.layers[1].output = zeros(T, input_shape)
        end

        model.default_input_shape = model.layers[last(model.D_range)].output_shape
        output_shape = (model.default_input_shape..., mini_batch)
        return initialize_layers!(model.layers, last(model.D_range) + 1, output_shape, mini_batch, T)
    end

    function init_targets!(model::GAN{F,T}, batch::Int) where {F,T}
        output_shape = model.layers[last(model.D_range)].output_shape
        output_shape in ((1,), (2,)) ||
            throw(DimensionMismatch("GAN training requires one or two discriminator outputs per example"))
        outputs = only(output_shape)
        if !isdefined(model, :real_target) || size(model.real_target) != (outputs, batch)
            model.real_target = zeros(T, outputs, batch)
            model.real_target[end, :] .= one(T)
            model.fake_target = zeros(T, outputs, batch)
            outputs == 2 && (model.fake_target[1, :] .= one(T))
        end
    end

    function activate_generator!(model::GAN; training::Bool=false)
        input = model.layers[1].output
        features = prod(model.layers[2].input_shape)
        for first in 1:features:length(input)
            sample = model.noise_generator()
            length(sample) == features || throw(DimensionMismatch("noise generator must return one sample"))
            copyto!(input, first, sample, 1, features)
        end
        return forward_layers!(model.layers, model.G_range, input, training)
    end

    function activate_discriminator!(model::GAN, data::Array{<:AbstractFloat}; training::Bool=false)
        return forward_layers!(model.layers, model.D_range, data, training)
    end

    function activate!(model::GAN; training::Bool=false)
        generated = activate_generator!(model; training)
        return activate_discriminator!(model, generated; training)
    end

    discriminator_step(::Tuple{}, offset) = ()
    discriminator_step(parameters, offset) = (2 * parameters[1] + offset, Base.tail(parameters)...)

    function update!(model::GAN, current_input_data::Array{<:AbstractFloat}, _current_output_data::Array{<:AbstractFloat},
                        loss_function, monitor, optimizer::String, α::Real, parameters...)
        init_targets!(model, size(current_input_data, ndims(current_input_data)))
        last_layer = model.layers[last(model.D_range)]
        real = activate_discriminator!(model, current_input_data; training=true)
        output_delta = output_buffer!(model.layers[end], real)
        model.loss += monitor.func(real, model.real_target)
        delta = DianoiaML.output_gradient!(output_delta, last_layer, model.real_target, loss_function)
        backward_layers!(model.layers, model.D_range, current_input_data, delta, optimizer, α, discriminator_step(parameters, -1), 1)

        generated = activate_generator!(model; training=true)
        fake = activate_discriminator!(model, generated; training=true)
        model.loss += monitor.func(fake, model.fake_target)
        delta = DianoiaML.output_gradient!(output_delta, last_layer, model.fake_target, loss_function)
        backward_layers!(model.layers, model.D_range, generated, delta, optimizer, α, discriminator_step(parameters, 0), 1)

        activate_discriminator!(model, generated; training=true)
        delta = DianoiaML.output_gradient!(output_delta, last_layer, model.real_target, loss_function)
        generator_delta = backward_layers!(model.layers, model.D_range, generated, delta, optimizer, α, parameters, 0)
        backward_layers!(model.layers, model.G_range, model.layers[1].output, generator_delta, optimizer, α, parameters, 1)
        return nothing
    end
end
