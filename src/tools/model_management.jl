function stack_from_name(name; precision=Float32)
    name == "Sequential" && return Sequential(; precision)
    name == "ResNet" && return ResNet(; precision)
    throw(ArgumentError("unknown nested network type $name"))
end

function save_stack_layers(file, model::AbstractLayerStack, prefix)
    for (i, layer) in enumerate(layer_children(model))
        save_configured_layer(file, layer, prefix * string(i))
    end
end

function save_configured_layer(file, layer, id)
    write_precision(file, id * "precision", layer)
    save_layer(layer, file, id)
end

function load_stack_layers!(model::AbstractLayerStack, file, prefix, count)
    for i in 1:count
        load_layer!(model, add_layer!, file, prefix * string(i))
    end
    return model
end

function save_layer(model::AbstractLayerStack, file, id::String)
    write(file, id, string(nameof(typeof(model))))
    write(file, id * "num_layer", model.num_layer)
    save_stack_layers(file, model, id * "layer")
end

function load_layer!(model, add_layer, file, id)
    name = read(file, id)
    if name in ("Sequential", "ResNet")
        block = stack_from_name(name; precision=read_precision(file, id * "precision"))
        load_stack_layers!(block, file, id * "layer", read(file, id * "num_layer"))
        add_layer(model, block)
        return
    end
    layer_type = getfield(@__MODULE__, Symbol(name))
    layer_module = getfield(@__MODULE__, Symbol(name * "M"))
    args = layer_module.get_args(file, id)
    if accepts_precision(layer_type)
        args = merge(args, (; precision=read_precision(file, id * "precision")))
    end
    if haskey(file, id * "activation_function")
        activation = getfield(@__MODULE__, Symbol(read(file, id * "activation_function")))
        add_layer(model, layer_type; args..., activation_function=activation)
    else
        add_layer(model, layer_type; args...)
    end
    layer = model.layers[model isa GAN ? model.num_Glayer + model.num_Dlayer + 1 : model.num_layer + 1]
    load_layer!(layer, file, id)
end

function save_Sequential(model::Sequential, path::String)
    validate_layer_tree!(model)
    h5open(path, "w") do file
        write_precision(file, "precision", model)
        write(file, "num_layer", model.num_layer)
        save_stack_layers(file, model, "")
    end
end

function load_Sequential(path::String)
    return h5open(path, "r") do file
        model = Sequential(; precision=read_precision(file, "precision"))
        load_stack_layers!(model, file, "", read(file, "num_layer"))
    end
end

function save_GAN(model::GAN, path::String)
    validate_layer_tree!(model)
    h5open(path, "w") do file
        write_precision(file, "precision", model)
        write(file, "num_Glayer", model.num_Glayer)
        write(file, "num_Dlayer", model.num_Dlayer)
        for i in 2:model.num_Glayer+model.num_Dlayer+1
            save_configured_layer(file, model.layers[i], string(i - 1))
        end
    end
end

function load_GAN(path::String, noise_generator)
    return h5open(path, "r") do file
        model = GAN(noise_generator; precision=read_precision(file, "precision"))
        num_Glayer = read(file, "num_Glayer")
        num_Dlayer = read(file, "num_Dlayer")
        for i in 1:num_Glayer
            load_layer!(model, model.add_Glayer, file, string(i))
        end
        for i in num_Glayer+1:num_Glayer+num_Dlayer
            load_layer!(model, model.add_Dlayer, file, string(i))
        end
        return model
    end
end
