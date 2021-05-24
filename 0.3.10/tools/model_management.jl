using HDF5

function incomplete_init(model::Sequential, name::String)
    if name=="Dense"
        model.add_layer(model, Dense; reload=true, input_size=0, layer_size=0, activation_function=None)
    elseif name=="Conv2D"
        model.add_layer(model, Conv2D; reload=true, input_filter=0, filter=0, input_size=0, input2D_size=(0,0), kernel_size=(0,0), activation_function=None)
    elseif name=="MaxPooling2D"
        model.add_layer(model, MaxPooling2D; reload=true, input_filter=0, input_size=0, input2D_size=(0,0), kernel_size=(0,0), activation_function=None)
    elseif name=="UpSampling2D"
        model.add_layer(model, UpSampling2D; reload=true, input_filter=0, input_size=0, input2D_size=(0,0), size=(0,0), activation_function=None)
    end
end

function save_Sequential(model::Sequential, path::String)
    h5open(path, "w") do file
        write(file, "num_layer", model.num_layer)
        for i in 2:model.num_layer+1
            model.layers[i].save_layer(model.layers[i], file, i-1)
        end
    end
end

function load_Sequential(path::String)
    list_of_ac_fun = [ReLU, Sigmoid, Softmax, Softmax_CEL, tanH, None]
    model = Sequential()

    h5open(path, "r") do file
        num_layer = read(file, "num_layer")
        for i in 1:num_layer
            layer_type = read(file, string(i))
            incomplete_init(model, layer_type)
            model.layers[end].load_layer(model.layers[end], file, i)
            ac_fun_type = read(file, string(i)*"activation_function")
            for f in list_of_ac_fun
                if f.get_name()==ac_fun_type
                    model.layers[end].activation_function = f
                    break
                end
            end
        end
    end
    return model
end

function save_GAN(model::GAN, path::String)
    h5open(path, "w") do file
        write(file, "num_Glayer", model.num_Glayer)
        write(file, "num_Dlayer", model.num_Dlayer)
        for i in model.G_range
            model.layers[i].save_layer(model.layers[i], file, i-1)
        end
        for i in model.D_range
            model.layers[i].save_layer(model.layers[i], file, i-1)
        end
    end
end

function load_GAN(path::String, noise_generator::Any)
    list_of_ac_fun = [ReLU, Sigmoid, Softmax, Softmax_CEL, tanH, None]
    model = GAN(noise_generator)

    h5open(path, "r") do file
        num_Glayer = read(file, "num_Glayer")
        num_Dlayer = read(file, "num_Dlayer")
        for i in 1:num_Glayer
            layer_type = read(file, string(i))
            incomplete_init(model, layer_type)
            model.temp_Glayers[end].load_layer(model.temp_Glayers[end], file, i)
            ac_fun_type = read(file, string(i)*"activation_function")
            for f in list_of_ac_fun
                if f.get_name()==ac_fun_type
                    model.temp_Glayers[end].activation_function = f
                    break
                end
            end
        end

        for i in num_Glayer+1:num_Glayer+num_Dlayer
            layer_type = read(file, string(i))
            incomplete_init(model, layer_type)
            model.temp_Dlayers[end].load_layer(model.temp_Dlayers[end], file, i)
            ac_fun_type = read(file, string(i)*"activation_function")
            for f in list_of_ac_fun
                if f.get_name()==ac_fun_type
                    model.temp_Dlayers[end].activation_function = f
                    break
                end
            end
        end
    end
    return model
end
