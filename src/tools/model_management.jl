using HDF5: h5open

function load_layer!(model, add_layer, file, id)
    name = read(file, id)
    layer_type = getfield(@__MODULE__, Symbol(name))
    layer_module = getfield(@__MODULE__, Symbol(name * "M"))
    args = layer_module.get_args(file, id)
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
    h5open(path, "w") do file
        write(file, "num_layer", model.num_layer)
        for i in 2:model.num_layer+1
            save_layer(model.layers[i], file, string(i - 1))
        end
    end
end

function load_Sequential(path::String)
    model = Sequential()
    h5open(path, "r") do file
        for i in 1:read(file, "num_layer")
            load_layer!(model, model.add_layer, file, string(i))
        end
    end
    return model
end

function save_GAN(model::GAN, path::String)
    h5open(path, "w") do file
        write(file, "num_Glayer", model.num_Glayer)
        write(file, "num_Dlayer", model.num_Dlayer)
        for i in 2:model.num_Glayer+model.num_Dlayer+1
            save_layer(model.layers[i], file, string(i - 1))
        end
    end
end

function load_GAN(path::String, noise_generator)
    model = GAN(noise_generator)
    h5open(path, "r") do file
        num_Glayer = read(file, "num_Glayer")
        num_Dlayer = read(file, "num_Dlayer")
        for i in 1:num_Glayer
            load_layer!(model, model.add_Glayer, file, string(i))
        end
        for i in num_Glayer+1:num_Glayer+num_Dlayer
            load_layer!(model, model.add_Dlayer, file, string(i))
        end
    end
    return model
end
