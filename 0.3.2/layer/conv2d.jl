module conv2d
    include("../tools/glorotuniform.jl")
    using .GlorotUniform
    include("../tools/avx_acc.jl")
    using .AVX_acc
    using LoopVectorization

    mutable struct Conv2D
        save_layer::Any
        load_layer::Any
        activator::Any
        get_PU::Any
        initializer::Any
        update_weights::Bool
        update_biases::Bool

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        filters::Array{Float32}
        weights::Array{Float32}
        weights_prop::Array{Int8}

        unit_size::Tuple{Int64, Int64}
        input2D_size::Tuple{Int64, Int64}
        step_x::Int64
        step_y::Int64

        Vdw::Array{Float32}
        Sdw::Array{Float32}

        conv_num_per_row::Int64
        conv_num_per_col::Int64

        value::Array{Float32}
        output::Array{Float32}
        propagation_units::Array{Float32}

        function Conv2D(;input_filter::Int64, filter::Int64, input_size::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, step_x::Int64=1, step_y::Int64=1, activation_function::Module, randomization::Bool=true, reload::Bool=false)
            if reload
                return new(save_Conv2D, load_Conv2D, activate_Conv2D, PU_Conv2D, init_Conv2D, true, false)
            end

            conv_num_per_row = (input2D_size[2]-kernel_size[2])÷step_x+1
            conv_num_per_col = (input2D_size[1]-kernel_size[1])÷step_y+1

            weights = zeros(Float32, (conv_num_per_row*conv_num_per_col*filter, input_size))
            weights_prop = zeros(Int8, size(weights))

            filter_size = (filter, input_filter, kernel_size[1], kernel_size[2])
            filters = !randomization ? zeros(Float32, filter_size) : Array{Float32}(GUMatrix(input_size, conv_num_per_row*conv_num_per_col*filter, filter_size))
            unit_size = (conv_num_per_row*conv_num_per_col, input_size÷input_filter)
            Threads.@threads for i in 0:filter-1
                for j in 0:input_filter-1
                    init_weights(i, j, weights, weights_prop, filters, step_x, step_y, input2D_size, kernel_size, unit_size, randomization, conv_num_per_row, conv_num_per_col)
                end
            end

            Vdw = zeros(Float32, size(weights))
            Sdw = zeros(Float32, size(weights))
            new(save_Conv2D, load_Conv2D, activate_Conv2D, PU_Conv2D, init_Conv2D, true, false, input_size, conv_num_per_row*conv_num_per_col*filter, activation_function, filters, weights, weights_prop, unit_size, input2D_size, step_x, step_y, Vdw, Sdw, conv_num_per_row, conv_num_per_col)
        end
    end

    function activate_Conv2D(layer::Conv2D, input::Array{Float32})
        #=
        Threads.@threads for i in 1:size(layer.filters, 1)
            for j in 1:size(layer.filters, 2)
                for k in 1:size(layer.filters, 3)
                    for l in 1:size(layer.filters, 4)
                        layer.filters[i,j,k,l] = flat_weights(i, j, layer.unit_size, layer.weights, (i-1)*layer.unit_size[1]+1, (k-1)*size(layer.filters, 4)+l, 0.0f0)
                    end
                end
            end
        end
        =#
        layer.value .= 0.0f0
        Conv2D_mul!(layer.value, layer.weights, input, layer.filters, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        # layer.value = layer.weights*input
        layer.output = layer.activation_function.func(layer.value)
    end

    function PU_Conv2D(layer::Conv2D, ∇biases::Array{Float32})
        layer.propagation_units .= 0.0f0
        Conv2D_PU!(layer.propagation_units, layer.weights, ∇biases, layer.filters, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        # layer.propagation_units = transpose(layer.weights)*∇biases
    end

    function init_Conv2D(layer::Conv2D, mini_batch::Int64)
        layer.value = zeros(Float32, (size(layer.weights, 1), mini_batch))
        layer.propagation_units = zeros(Float32, (size(layer.weights, 2), mini_batch))
    end

    function Conv2D_mul!(value::Array{Float32}, weights::Array{Float32}, input::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(input, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        value[x*unit_size[1]+i+1, b] += weights[x*unit_size[1]+i+1, y*unit_size[2]+index+k] * input[y*unit_size[2]+index+k, b]
                    end
                    index += input2D_size[2]
                end
            end
        end
    end

    function Conv2D_PU!(propagation_units::Array{Float32}, weights::Array{Float32}, ∇biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        propagation_units[y*unit_size[2]+index+k, b] += weights[x*unit_size[1]+i+1, y*unit_size[2]+index+k] * ∇biases[x*unit_size[1]+i+1, b]
                    end
                    index += input2D_size[2]
                end
            end
        end
    end

    function init_weights(x::Int64, y::Int64, weights::Array{Float32}, weights_prop::Array{Int8}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, randomization::Bool, conv_num_per_row::Int64, conv_num_per_col::Int64)
        for i in 0:unit_size[1]-1
            index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
            for j in 1:kernel_size[1]
                for k in 1:kernel_size[2]
                    if randomization
                        weights[x*unit_size[1]+i+1, y*unit_size[2]+index+k] = filters[x+1,y+1,j,k]
                    end
                    weights_prop[x*unit_size[1]+i+1, y*unit_size[2]+index+k] = Int8(1)
                end
                index += input2D_size[2]
            end
        end
    end

    function flat_weights(x::Int64, y::Int64, unit_size::Tuple{Int64, Int64}, weights::Array{Float32}, layer_index::Int64, kernel_index::Int64, s::Float32)
        counter = 0
        for i in (y-1)*unit_size[2]+1:y*unit_size[2]
            if weights[layer_index,i]!=0
                counter += 1
                if counter==kernel_index
                    s += weights[layer_index,i]
                end
            end
        end
        if layer_index<x*unit_size[1]
            return flat_weights(x, y, unit_size, weights, layer_index+1, kernel_index, s)
        else
            s = s/layer_index
            for i in (x-1)*unit_size[1]+1:x*unit_size[1]
                counter = 0
                for l in (y-1)*unit_size[2]+1:y*unit_size[2]
                    if weights[i,l]!=0
                        counter += 1
                        if counter==kernel_index
                            weights[i,l] = s
                            return s
                        end
                    end
                end
            end
        end
    end

    function save_Conv2D(layer::Conv2D, file::Any, id::Int64)
        write(file, string(id), "Conv2D")
        Threads.@threads for i in axes(layer.filters, 1)
            for j in axes(layer.filters, 2)
                for k in axes(layer.filters, 3)
                    for l in axes(layer.filters, 4)
                        layer.filters[i,j,k,l] = flat_weights(i, j, layer.unit_size, layer.weights, (i-1)*layer.unit_size[1]+1, (k-1)*size(layer.filters, 4)+l, 0.0f0)
                    end
                end
            end
        end
        write(file, string(id)*"filters", layer.filters)
        write(file, string(id)*"weights", layer.weights)
        write(file, string(id)*"weights_prop", layer.weights_prop)
        write(file, string(id)*"unit_size", collect(layer.unit_size))
        write(file, string(id)*"input2D_size", collect(layer.input2D_size))
        write(file, string(id)*"step_x", layer.step_x)
        write(file, string(id)*"step_y", layer.step_y)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_Conv2D(layer::Conv2D, file::Any, id::Int64)
        layer.filters = read(file, string(id)*"filters")
        layer.weights = read(file, string(id)*"weights")
        layer.weights_prop = read(file, string(id)*"weights_prop")
        layer.unit_size = Tuple(read(file, string(id)*"unit_size"))
        layer.input2D_size = Tuple(read(file, string(id)*"input2D_size"))
        layer.step_x = read(file, string(id)*"step_x")
        layer.step_y = read(file, string(id)*"step_y")
        layer.input_size = size(layer.weights, 2)
        layer.layer_size = size(layer.weights, 1)
        layer.Vdw = zeros(Float32, size(layer.weights))
        layer.Sdw = zeros(Float32, size(layer.weights))
        kernel_size = (size(layer.filters,3), size(layer.filters,4))
        layer.conv_num_per_row = (layer.input2D_size[2]-kernel_size[2])÷layer.step_x+1
        layer.conv_num_per_col = (layer.input2D_size[1]-kernel_size[1])÷layer.step_y+1
    end
end

# Source: https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
