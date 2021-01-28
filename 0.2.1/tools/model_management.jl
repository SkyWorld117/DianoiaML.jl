#using Pkg
#Pkg.add("HDF5")
using HDF5

function save_sequential(sequential::Any, path::String)
    h5open(path, "w") do file
        write(file, "num_layer", sequential.num_layer)
        for i in 2:sequential.num_layer+1
            write(file, string(i-1), string(typeof(sequential.layers[i]))) # layer_type
            write(file, string(i-1)*"weights", sequential.layers[i].weights) # weights
            write(file, string(i-1)*"biases", sequential.layers[i].biases) # biases
            write(file, string(i-1)*"activation_function", sequential.layers[i].activation_function.get_name()) # activation_function
        end
    end
end

function load_sequential(path)
    list_of_ac_fun = [ReLU, Sigmoid, Softmax, tanH, None]
    list_of_layers = [Dense, Convolutional_2D]
    model = Sequential()

    h5open(path, "r") do file
        num_layer = read(file, "num_layer")
        for i in 1:num_layer
            layer_type = read(file, string(i))
            for j in list_of_layers
                if string(j)==layer_type
                    model.add_layer(model, j(reload=true, input_size=0, layer_size=0, randomization=false, activation_function=list_of_ac_fun[1]))
                    break
                end
            end
            model.layers[i+1].weights = read(file, string(i)*"weights")
            model.layers[i+1].biases = read(file, string(i)*"biases")
            ac_fun_type = read(file, string(i)*"activation_function")
            for j in list_of_ac_fun
                if j.get_name()==ac_fun_type
                    model.layers[i+1].activation_function = j
                    break
                end
            end
            model.layers[i+1].input_size = size(model.layers[i+1].weights, 2)
            model.layers[i+1].layer_size = size(model.layers[i+1].weights, 1)
        end
    end
    push!(model.layers, Hidden_Output_Layer(Float32[]))
    return model
end
