using .Threads

module convolutional_2d
    mutable struct Convolutional_2D
        save_layer::Any
        load_layer::Any
        activator::Any
        get_PU::Any
        update_weights::Bool
        update_biases::Bool

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        kernel::Array{Float32}
        weights::Array{Float32}
        weights_prop::Array{Int8}

        input2D_size::Tuple{Int64, Int64}
        step_x::Int64
        step_y::Int64

        value::Array{Float32}
        output::Array{Float32}
        propagation_units::Array{Float32}

        Vdw::Array{Float32}
        Sdw::Array{Float32}

        function Convolutional_2D(;input_size::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, step_x::Int64=1, step_y::Int64=1, activation_function::Module, randomization::Bool=true, reload::Bool=false)
            if reload
                return new(save_Conv2D, load_Conv2D, activate_Conv2D, PU_Conv2D, true, false)
            end

            conv_num_per_row = (input2D_size[2]-kernel_size[2])÷step_x+1
            conv_num_per_col = (input2D_size[1]-kernel_size[1])÷step_y+1

            weights = zeros(Float32, (conv_num_per_row*conv_num_per_col, input_size))
            weights_prop = zeros(Int8, size(weights))

            kernel = !randomization ? zeros(Float32, kernel_size) : 0.1*rand(Float32, kernel_size)
            for i in 0:size(weights, 1)-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    Threads.@threads for k in 1:kernel_size[2]
                        if randomization
                            weights[i+1,index+k] = kernel[j,k]
                        end
                        weights_prop[i+1,index+k] = Int8(1)
                    end
                    index += input2D_size[2]
                end
            end
            new(save_Conv2D, load_Conv2D, activate_Conv2D, PU_Conv2D, true, false, input_size, conv_num_per_row*conv_num_per_col, activation_function, kernel, weights, weights_prop, input2D_size, step_x, step_y)
        end
    end

    function activate_Conv2D(layer::Convolutional_2D, input::Array{Float32})
        Threads.@threads for i in 1:length(layer.kernel)
            layer.kernel[i] = flat_weights(layer.weights, 1, i, 0.0f0)
        end

        layer.value = layer.weights*input
        layer.output = layer.activation_function.func(layer.value)

        layer.Vdw = zeros(Float32, size(layer.weights))
        layer.Sdw = zeros(Float32, size(layer.weights))
    end

    function PU_Conv2D(weights::Array{Float32}, ∇biases::Array{Float32})
        return transpose(weights)*∇biases
    end

    function flat_weights(weights::Array{Float32}, layer_index::Int64, kernel_index::Int64, s::Float32)
        counter = 0
        for i in 1:size(weights, 2)
            if weights[layer_index,i]!=0
                counter += 1
                if counter==kernel_index
                    s += weights[layer_index,i]
                end
            end
        end
        if layer_index<size(weights, 1)
            return flat_weights(weights, layer_index+1, kernel_index, s)
        else
            s = s/layer_index
            for i in 1:size(weights, 1)
                counter = 0
                for l in 1:size(weights, 2)
                    if weights[i,l]!=0
                        counter += 1
                        if counter==kernel_index
                            weights[i,l] = s
                            break
                        end
                    end
                end
            end
        end
        return s
    end

    function save_Conv2D(layer::Convolutional_2D, file::Any, id::Int64)
        write(file, string(id), "Convolutional_2D")
        Threads.@threads for i in 1:length(layer.kernel)
            layer.kernel[i] = flat_weights(layer.weights, 1, i, 0.0f0)
        end
        write(file, string(id)*"kernel", layer.kernel)
        write(file, string(id)*"weights", layer.weights)
        write(file, string(id)*"weights_prop", layer.weights_prop)
        write(file, string(id)*"input2D_size", collect(layer.input2D_size))
        write(file, string(id)*"step_x", layer.step_x)
        write(file, string(id)*"step_y", layer.step_y)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_Conv2D(layer::Convolutional_2D, file::Any, id::Int64)
        layer.kernel = read(file, string(id)*"kernel")
        layer.weights = read(file, string(id)*"weights")
        layer.weights_prop = read(file, string(id)*"weights_prop")
        layer.input2D_size = Tuple(read(file, string(id)*"input2D_size"))
        layer.step_x = read(file, string(id)*"step_x")
        layer.step_y = read(file, string(id)*"step_y")
        layer.input_size = size(layer.weights, 2)
        layer.layer_size = size(layer.weights, 1)
    end
end

# Source: https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
