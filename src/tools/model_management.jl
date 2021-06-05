using HDF5

function save_Sequential(model::Sequential, path::String)
    h5open(path, "w") do file
        write(file, "num_layer", model.num_layer)
        for i in 2:model.num_layer+1
            model.layers[i].save_layer(model.layers[i], file, string(i-1))
        end
    end
end

function load_Sequential(path::String)
    model = Sequential()

    h5open(path, "r") do file
        num_layer = read(file, "num_layer")
        for i in 1:num_layer
            s = read(file, string(i))
            layer = Symbol(s)
            mod = Symbol(s*"M")
            args = eval(mod).get_args(file, string(i))
            try
                activation_function = Symbol(read(file, string(i)*"activation_function"))
                model.add_layer(model, eval(layer); args..., activation_function=eval(activation_function))
            catch KeyError
                model.add_layer(model, eval(layer); args...)
            end
            model.layers[end].load_layer(model.layers[end], file, string(i))
        end
    end
    return model
end

function save_GAN(model::GAN, path::String)
    h5open(path, "w") do file
        write(file, "num_Glayer", model.num_Glayer)
        write(file, "num_Dlayer", model.num_Dlayer)
        for i in model.G_range
            model.layers[i].save_layer(model.layers[i], file, string(i-1))
        end
        for i in model.D_range
            model.layers[i].save_layer(model.layers[i], file, string(i-1))
        end
    end
end

function load_GAN(path::String, noise_generator::Any)
    model = GAN(noise_generator)

    h5open(path, "r") do file
        num_Glayer = read(file, "num_Glayer")
        num_Dlayer = read(file, "num_Dlayer")
        for i in 1:num_Glayer
            s = read(file, string(i))
            layer = Symbol(s)
            mod = Symbol(s*"M")
            args = eval(mod).get_args(file, string(i))
            try
                activation_function = Symbol(read(file, string(i)*"activation_function"))
                model.add_Glayer(model, eval(layer); args..., activation_function=eval(activation_function))
            catch KeyError
                model.add_Glayer(model, eval(layer); args...)
            end
            model.layers[end].load_layer(model.layers[end], file, string(i))
        end

        for i in num_Glayer+1:num_Glayer+num_Dlayer
            s = read(file, string(i))
            layer = Symbol(s)
            mod = Symbol(s*"M")
            args = eval(mod).get_args(file, string(i))
            try
                activation_function = Symbol(read(file, string(i)*"activation_function"))
                model.add_Dlayer(model, eval(layer); args..., activation_function=eval(activation_function))
            catch KeyError
                model.add_Dlayer(model, eval(layer); args...)
            end
            model.layers[end].load_layer(model.layers[end], file, string(i))
        end
    end
    return model
end
