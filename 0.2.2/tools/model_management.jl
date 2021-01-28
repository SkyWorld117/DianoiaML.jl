#using Pkg
#Pkg.add("HDF5")
using HDF5

function save_Sequential(model::Sequential, path::String)
    h5open(path, "w") do file
        write(file, "num_layer", model.num_layer)
        for i in 2:model.num_layer+1
            model.layers[i].save_layer(model.layers[i], file, i-1)
        end
    end
end

function load_Sequential(path::String)
    list_of_ac_fun = [ReLU, Sigmoid, Softmax, tanH, None]
    model = Sequential()

    h5open(path, "r") do file
        num_layer = read(file, "num_layer")
        for i in 1:num_layer
            layer_type = read(file, string(i))
            model.add_layer(model, incomplete_init(layer_type))
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
    push!(model.layers, Hidden_Output_Layer(Float32[]))
    return model
end

function incomplete_init(name::String)
    if name=="Dense"
        return Dense(reload=true, input_size=0, layer_size=0, randomization=false, activation_function=None)
    elseif name=="Convolutional_2D"
        return Convolutional_2D(reload=true, input_size=0, input2D_size=(0,0), kernel_size=(0,0), activation_function=None)
    elseif name=="MaxPooling_2D"
        return MaxPooling_2D(reload=true, input_size=0, input2D_size=(0,0), kernel_size=(0,0), activation_function=None)
    end
end
