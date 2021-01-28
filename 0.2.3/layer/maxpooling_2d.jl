using .Threads

module maxpooling_2d
    mutable struct MaxPooling_2D
        save_layer::Any
        load_layer::Any
        activator::Any
        get_PU::Any
        update_weights::Bool
        update_biases::Bool

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        unit_size::Tuple{Int64, Int64}
        input2D_size::Tuple{Int64, Int64}
        kernel_size::Tuple{Int64, Int64}
        step_x::Int64
        step_y::Int64

        weights::Array{Int8}
        value::Array{Float32}
        output::Array{Float32}
        propagation_units::Array{Float32}

        function MaxPooling_2D(;input_filter::Int64, input_size::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, step_x::Int64=kernel_size[2], step_y::Int64=kernel_size[1], activation_function::Module, reload::Bool=false)
            if reload
                return new(save_MaxPooling2D, load_MaxPooling2D, activate_MaxPooling2D, PU_MaxPooling2D, false, false)
            end

            conv_num_per_row = (input2D_size[2]-kernel_size[2])÷step_x+1
            conv_num_per_col = (input2D_size[1]-kernel_size[1])÷step_y+1
            unit_size = (conv_num_per_row*conv_num_per_col, input_size÷input_filter)
            new(save_MaxPooling2D, load_MaxPooling2D, activate_MaxPooling2D, PU_MaxPooling2D, false, false, input_size, conv_num_per_row*conv_num_per_col*input_filter, activation_function, unit_size, input2D_size, kernel_size, step_x, step_y)
        end
    end

    function activate_MaxPooling2D(layer::MaxPooling_2D, input::Array{Float32})
        batch_size = size(input, 2)
        layer.weights = zeros(Int8, (layer.layer_size, layer.input_size, batch_size))
        layer.value = zeros(Float32, (layer.layer_size, batch_size))

        conv_num_per_row = (layer.input2D_size[2]-layer.kernel_size[2])÷layer.step_x+1
        conv_num_per_col = (layer.input2D_size[1]-layer.kernel_size[1])÷layer.step_y+1
        Threads.@threads for b in 1:batch_size
            for i in 1:layer.input_size÷layer.unit_size[2]
                create_weights(layer, input, b, i, conv_num_per_row, conv_num_per_col)
            end
            layer.value[:,b] = layer.weights[:,:,b]*input[:,b]
        end
        layer.output = layer.activation_function.func(layer.value)
    end

    function PU_MaxPooling2D(weights::Array{Int8}, ∇biases::Array{Float32})
        batch_size = size(∇biases, 2)
        propagation_units = zeros(Float32, (size(weights, 2),batch_size))
        Threads.@threads for b in 1:batch_size
            propagation_units[:,b] = transpose(weights[:,:,b])*∇biases[:,b]
        end
        return propagation_units
    end

    function create_weights(layer::MaxPooling_2D, input::Array{Float32}, b::Int64, i::Int64, conv_num_per_row::Int64, conv_num_per_col::Int64)
        for l in 0:layer.unit_size[1]-1
            index = layer.step_x*(l%conv_num_per_row) + layer.input2D_size[2]*layer.step_y*(l÷conv_num_per_row)
            max_value = -Inf
            register = 0
            for j in 1:layer.kernel_size[1]
                for k in 1:layer.kernel_size[2]
                    if input[(i-1)*layer.unit_size[2]+index+k,b]>max_value
                        max_value = input[index+k,b]
                        register = index+k
                    end
                end
                index += layer.input2D_size[2]
            end
            layer.weights[(i-1)*layer.unit_size[1]+l+1, (i-1)*layer.unit_size[2]+register, b] = Int8(1)
        end
    end

    function save_MaxPooling2D(layer::MaxPooling_2D, file::Any, id::Int64)
        write(file, string(id), "MaxPooling_2D")
        write(file, string(id)*"layer_size", layer.layer_size)
        write(file, string(id)*"unit_size", collect(layer.unit_size))
        write(file, string(id)*"input2D_size", collect(layer.input2D_size))
        write(file, string(id)*"kernel_size", collect(layer.kernel_size))
        write(file, string(id)*"step_x", layer.step_x)
        write(file, string(id)*"step_y", layer.step_y)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_MaxPooling2D(layer::MaxPooling_2D, file::Any, id::Int64)
        layer.unit_size = Tuple(read(file, string(id)*"unit_size"))
        layer.input2D_size = Tuple(read(file, string(id)*"input2D_size"))
        layer.input_size = layer.input2D_size[1]*layer.input2D_size[2]
        layer.layer_size = read(file, string(id)*"layer_size")
        layer.kernel_size = Tuple(read(file, string(id)*"kernel_size"))
        layer.step_x = read(file, string(id)*"step_x")
        layer.step_y = read(file, string(id)*"step_y")
    end
end
