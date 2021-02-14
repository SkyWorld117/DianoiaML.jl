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
        initializer::Any
        updater::Any

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        filters::Array{Float32}

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
                return new(save_Conv2D, load_Conv2D, activate_Conv2D, init_Conv2D, update_Conv2D)
            end

            conv_num_per_row = (input2D_size[2]-kernel_size[2])÷step_x+1
            conv_num_per_col = (input2D_size[1]-kernel_size[1])÷step_y+1

            filter_size = (filter, input_filter, kernel_size[1], kernel_size[2])
            filters = !randomization ? zeros(Float32, filter_size) : Array{Float32}(GUMatrix(input_size, conv_num_per_row*conv_num_per_col*filter, filter_size))
            unit_size = (conv_num_per_row*conv_num_per_col, input_size÷input_filter)

            Vdw = zeros(Float32, filter_size)
            Sdw = zeros(Float32, filter_size)
            new(save_Conv2D, load_Conv2D, activate_Conv2D, init_Conv2D, update_Conv2D, input_size, conv_num_per_row*conv_num_per_col*filter, activation_function, filters, unit_size, input2D_size, step_x, step_y, Vdw, Sdw, conv_num_per_row, conv_num_per_col)
        end
    end

    function init_Conv2D(layer::Conv2D, mini_batch::Int64)
        layer.value = zeros(Float32, (layer.layer_size, mini_batch))
        layer.propagation_units = zeros(Float32, (layer.input_size, mini_batch))
    end

    function activate_Conv2D(layer::Conv2D, input::Array{Float32})
        layer.value .= 0.0f0
        Conv2D_mul!(layer.value, input, layer.filters, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        # layer.value = layer.weights*input
        layer.output = layer.activation_function.func(layer.value)
    end

    function update_Conv2D(layer::Conv2D, optimizer::String, Last_Layer_output::Array{Float32}, Next_Layer_propagation_units::Array{Float32}, α::Float64, parameters...)
        ∇biases = layer.activation_function.get_∇biases(layer.value, Next_Layer_propagation_units)

        layer.propagation_units .= 0.0f0
        Conv2D_PU!(layer.propagation_units, ∇biases, layer.filters, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        # layer.propagation_units = transpose(layer.weights)*∇biases

        if optimizer=="SGD"
            SGD!(α, Last_Layer_output, ∇biases, layer.filters, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
            # Current_Layer.weights -= ∇biases*transpose(Last_Layer.output).*α.*Current_Layer.weights_prop
        elseif optimizer=="Minibatch_GD"
            Minibatch_GD!(α, Last_Layer_output, ∇biases, layer.filters, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        elseif optimizer=="Adam"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]
            Adam!(α, t, β₁, β₂, ϵ, Last_Layer_output, ∇biases, layer.filters, layer.Vdw, layer.Sdw, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        elseif optimizer=="AdaBelief"
            AdaBelief!(α, t, β₁, β₂, ϵ, Last_Layer_output, ∇biases, layer.filters, layer.Vdw, layer.Sdw, layer.step_x, layer.step_y, layer.input2D_size, (size(layer.filters,3), size(layer.filters,4)), layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
        end
    end

    function Conv2D_mul!(value::Array{Float32}, input::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(input, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        value[x*unit_size[1]+i+1, b] += filters[x+1,y+1,j,k] * input[y*unit_size[2]+index+k, b]
                    end
                    index += input2D_size[2]
                end
            end
        end
    end

    function Conv2D_PU!(propagation_units::Array{Float32}, ∇biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        propagation_units[y*unit_size[2]+index+k, b] += filters[x+1,y+1,j,k] * ∇biases[x*unit_size[1]+i+1, b]
                    end
                    index += input2D_size[2]
                end
            end
        end
    end

    function SGD!(α::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        filters[x+1,y+1,j,k] -= α/unit_size[1]*∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]
                    end
                    index += input2D_size[2]
                end
            end
        end
    end

    function Minibatch_GD!(α::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        c = 0.0f0
                        for b in axes(∇biases, 2)
                            c += ∇biases[x*unit_size[1]+i+1,b]*Last_Layer_output[y*unit_size[2]+index+k,b]
                        end
                        filters[x+1,y+1,j,k] -= c*α/unit_size[1]
                    end
                    index += input2D_size[2]
                end
            end
        end
    end

    function Adam!(α::Float64, t::Int64, β₁::Float64, β₂::Float64, ϵ::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, filters::Array{Float32}, Vdw::Array{Float32}, Sdw::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            Vdw[a,b,c,d] *= β₁
            Sdw[a,b,c,d] *= β₂
        end
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        Vdw[x+1,y+1,j,k] += ∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]*(1-β₁)/unit_size[1]
                        Sdw[x+1,y+1,j,k] += (∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1])^2*(1-β₂)/unit_size[1]
                    end
                    index += input2D_size[2]
                end
            end
        end
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            filters[a,b,c,d] -= α*(Vdw[a,b,c,d]/(1-β₁^t))/(sqrt(Sdw[a,b,c,d]/(1-β₂^t))+ϵ)
        end
    end

    function AdaBelief!(α::Float64, t::Int64, β₁::Float64, β₂::Float64, ϵ::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, filters::Array{Float32}, Vdw::Array{Float32}, Sdw::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            Vdw[a,b,c,d] *= β₁
            Sdw[a,b,c,d] *= β₂
        end
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        Vdw[x+1,y+1,j,k] += ∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]*(1-β₁)/unit_size[1]
                        Sdw[x+1,y+1,j,k] += (∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]-Vdw[x+1,y+1,j,k])^2*(1-β₂)/unit_size[1]
                    end
                    index += input2D_size[2]
                end
            end
        end
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            filters[a,b,c,d] -= α*(Vdw[a,b,c,d]/(1-β₁^t))/(sqrt(Sdw[a,b,c,d]/(1-β₂^t))+ϵ)
        end
    end

    function init_weights(x::Int64, y::Int64, weights::Array{Float32}, weights_prop::Array{Int8}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, randomization::Bool, conv_num_per_row::Int64, conv_num_per_col::Int64)
        @avx for i in 0:unit_size[1]-1
            index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
            for j in 1:kernel_size[1]
                for k in 1:kernel_size[2]
                    weights[x*unit_size[1]+i+1, y*unit_size[2]+index+k] = filters[x+1,y+1,j,k]
                    weights_prop[x*unit_size[1]+i+1, y*unit_size[2]+index+k] = Int8(1)
                end
                index += input2D_size[2]
            end
        end
    end

    function save_Conv2D(layer::Conv2D, file::Any, id::Int64)
        write(file, string(id), "Conv2D")
        write(file, string(id)*"input_size", layer.input_size)
        write(file, string(id)*"filters", layer.filters)
        write(file, string(id)*"unit_size", collect(layer.unit_size))
        write(file, string(id)*"input2D_size", collect(layer.input2D_size))
        write(file, string(id)*"step_x", layer.step_x)
        write(file, string(id)*"step_y", layer.step_y)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_Conv2D(layer::Conv2D, file::Any, id::Int64)
        layer.input_size = read(file, string(id)*"input_size")
        layer.filters = read(file, string(id)*"filters")
        layer.unit_size = Tuple(read(file, string(id)*"unit_size"))
        layer.input2D_size = Tuple(read(file, string(id)*"input2D_size"))
        layer.step_x = read(file, string(id)*"step_x")
        layer.step_y = read(file, string(id)*"step_y")
        layer.Vdw = zeros(Float32, size(layer.filters))
        layer.Sdw = zeros(Float32, size(layer.filters))
        kernel_size = (size(layer.filters,3), size(layer.filters,4))
        layer.conv_num_per_row = (layer.input2D_size[2]-kernel_size[2])÷layer.step_x+1
        layer.conv_num_per_col = (layer.input2D_size[1]-kernel_size[1])÷layer.step_y+1
        layer.layer_size = layer.conv_num_per_row*layer.conv_num_per_col*size(layer.filters, 1)
    end
end

# Source: https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
