module conv2d
    include("../tools/glorotuniform.jl")
    using .GlorotUniform
    using LoopVectorization

    mutable struct Conv2D
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_size::Int64
        layer_size::Int64
        kernel_size::Tuple{Int64, Int64}
        activation_function::Module

        filters::Array{Float32}
        biases::Array{Float32}

        unit_size::Tuple{Int64, Int64}
        padding::Int64
        input2D_size::Tuple{Int64, Int64}
        original_input2D_size::Tuple{Int64, Int64}
        step_x::Int64
        step_y::Int64

        Vdw::Array{Float32}
        Sdw::Array{Float32}
        Vdb::Array{Float32}
        Sdb::Array{Float32}

        conv_num_per_row::Int64
        conv_num_per_col::Int64

        padding_input::Array{Float32}
        value::Array{Float32}
        output::Array{Float32}
        ∇biases::Array{Float32}
        pre_propagation_units::Array{Float32}
        propagation_units::Array{Float32}
        original_unit::Int64
        input_filter::Int64
        args::Tuple

        function Conv2D(;input_filter::Int64, filter::Int64, input_size::Int64, input2D_size::Tuple{Int64, Int64}, padding::Int64=0, kernel_size::Tuple{Int64, Int64}, step_x::Int64=1, step_y::Int64=1, activation_function::Module, randomization::Bool=true, reload::Bool=false)
            if reload
                return new(save_Conv2D, load_Conv2D, activate_Conv2D, init_Conv2D, update_Conv2D)
            end

            original_input2D_size = input2D_size
            if padding!=0
                input2D_size = (input2D_size[1]+2*padding, input2D_size[2]+2*padding)
                input_size = input2D_size[1]*input2D_size[2]*input_filter
            end

            conv_num_per_row = (input2D_size[2]-kernel_size[2])÷step_x+1
            conv_num_per_col = (input2D_size[1]-kernel_size[1])÷step_y+1

            filter_size = (filter, input_filter, kernel_size[1], kernel_size[2])
            filters = !randomization ? zeros(Float32, filter_size) : Array{Float32}(GUMatrix(input_size, conv_num_per_row*conv_num_per_col*filter, filter_size))
            biases = !randomization ? zeros(Float32, filter) : Array{Float32}(GUMatrix(input_size, conv_num_per_row*conv_num_per_col*filter, filter))
            unit_size = (conv_num_per_row*conv_num_per_col, input_size÷input_filter)

            Vdw = zeros(Float32, filter_size)
            Sdw = zeros(Float32, filter_size)
            Vdb = zeros(Float32, filter)
            Sdb = zeros(Float32, filter)
            new(save_Conv2D, load_Conv2D, activate_Conv2D, init_Conv2D, update_Conv2D, input_size, conv_num_per_row*conv_num_per_col*filter, kernel_size, activation_function, filters, biases, unit_size, padding, input2D_size, original_input2D_size, step_x, step_y, Vdw, Sdw, Vdb, Sdb, conv_num_per_row, conv_num_per_col)
        end
    end

    function init_Conv2D(layer::Conv2D, mini_batch::Int64)
        layer.value = zeros(Float32, (layer.layer_size, mini_batch))
        layer.∇biases = zeros(Float32, (layer.layer_size, mini_batch))
        if layer.padding!=0
            layer.padding_input = zeros(Float32, (layer.input_size, mini_batch))
        end
        layer.pre_propagation_units = zeros(Float32, (layer.input_size, mini_batch))
        layer.original_unit = layer.original_input2D_size[1]*layer.original_input2D_size[2]
        layer.input_filter = size(layer.filters, 2)
        original_input_size = layer.original_unit*layer.input_filter
        layer.propagation_units = zeros(Float32, (original_input_size, mini_batch))

        layer.args = (layer.filters, layer.step_x, layer.step_y, layer.input2D_size, layer.kernel_size, layer.unit_size, layer.conv_num_per_row, layer.conv_num_per_col)
    end

    function activate_Conv2D(layer::Conv2D, input::Array{Float32})
        @avx for i in axes(layer.value, 1), j in axes(layer.value, 2)
            layer.value[i,j] = 0.0f0
        end
        if layer.padding!=0
            Conv2D_padding!(layer.padding_input, input, layer.input_filter, layer.padding, layer.input2D_size, layer.unit_size, size(layer.value, 2), layer.original_unit, layer.original_input2D_size)
        else
            layer.padding_input = input
        end
        Conv2D_mul_add!(layer.value, layer.padding_input, layer.biases, layer.args...)
        # layer.value = layer.weights*input
        layer.output = layer.activation_function.func(layer.value)
    end

    function update_Conv2D(layer::Conv2D, optimizer::String, Last_Layer_output::Array{Float32}, Next_Layer_propagation_units::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        layer.activation_function.get_∇biases!(layer.∇biases, layer.value, Next_Layer_propagation_units)

        if layer.padding!=0
            layer.pre_propagation_units .= 0.0f0
        end
        layer.propagation_units .= 0.0f0
        Conv2D_PU!(layer.pre_propagation_units, layer.propagation_units, layer.∇biases, layer.padding, layer.args..., layer.input_filter, layer.original_input2D_size, layer.original_unit)
        # layer.propagation_units = transpose(layer.weights)*∇biases

        if optimizer=="SGD"
            SGD!(α, layer.padding_input, layer.∇biases, layer.biases, layer.args..., direction)
            # Current_Layer.weights -= ∇biases*transpose(Last_Layer.output).*α.*Current_Layer.weights_prop
        elseif optimizer=="Minibatch_GD"
            Minibatch_GD!(α, layer.padding_input, layer.∇biases, layer.biases, layer.args..., direction)
        elseif optimizer=="Adam"
            Adam!(α, parameters..., layer.padding_input, layer.∇biases, layer.biases, layer.Vdw, layer.Sdw, layer.Vdb, layer.Sdb, layer.args..., direction)
        elseif optimizer=="AdaBelief"
            AdaBelief!(α, parameters..., layer.padding_input, layer.∇biases, layer.biases, layer.Vdw, layer.Sdw, layer.Vdb, layer.Sdb, layer.args..., direction)
        end
    end

    function Conv2D_padding!(padding_input::Array{Float32}, input::Array{Float32}, input_filter::Int64, padding::Int64, input2D_size::Tuple, unit_size::Tuple{Int64, Int64}, batch_size::Int64, original_unit::Int64, original_input2D_size::Tuple{Int64, Int64})
        @avx for b in 1:batch_size, f in 0:input_filter-1, i in 1:original_input2D_size[1], j in 0:original_input2D_size[2]-1
            padding_input[f*unit_size[2]+(padding+j)*input2D_size[1]+padding+i, b] = input[f*original_unit+j*original_input2D_size[1]+i, b]
        end
    end

    function Conv2D_mul_add!(value::Array{Float32}, input::Array{Float32}, biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64)
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

        @avx for x in 0:size(filters, 1)-1, y in 1:unit_size[1], b in axes(value, 2)
            value[x*unit_size[1]+y, b] += biases[x+1]
        end
    end

    function Conv2D_PU!(pre_propagation_units::Array{Float32}, propagation_units::Array{Float32}, ∇biases::Array{Float32}, padding::Int64, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64, input_filter::Int64, original_input2D_size::Tuple, original_unit::Int64)
        if padding!=0
            @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1, b in axes(∇biases, 2)
                for i in 0:unit_size[1]-1
                    index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                    for j in 1:kernel_size[1]
                        for k in 1:kernel_size[2]
                            pre_propagation_units[y*unit_size[2]+index+k, b] += filters[x+1,y+1,j,k] * ∇biases[x*unit_size[1]+i+1, b]
                        end
                        index += input2D_size[2]
                    end
                end
            end
            @avx for b in axes(propagation_units, 2), f in 0:input_filter-1, i in 1:original_input2D_size[1], j in 0:original_input2D_size[2]-1
                propagation_units[f*original_unit+j*original_input2D_size[1]+i, b] = pre_propagation_units[f*unit_size[2]+(padding+j)*input2D_size[1]+padding+i, b]
            end
        else
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
    end

    function SGD!(α::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64, direction::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        filters[x+1,y+1,j,k] -= α*∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]*direction
                    end
                    index += input2D_size[2]
                end
            end
        end

        @avx for x in 0:size(filters, 1)-1, y in 1:unit_size[1]
            biases[x+1] -= α*∇biases[x*unit_size[1]+y, 1]*direction
        end
    end

    function Minibatch_GD!(α::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, biases::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64, direction::Int64)
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        c = 0.0f0
                        for b in axes(∇biases, 2)
                            c += ∇biases[x*unit_size[1]+i+1,b]*Last_Layer_output[y*unit_size[2]+index+k,b]
                        end
                        filters[x+1,y+1,j,k] -= c*α*direction/size(∇biases, 2)
                    end
                    index += input2D_size[2]
                end
            end
        end

        @avx for x in 0:size(filters, 1)-1, y in 1:unit_size[1], b in axes(∇biases, 2)
            biases[x+1] -= α*∇biases[x*unit_size[1]+y, b]*direction/size(∇biases, 2)
        end
    end

    function Adam!(α::Float64, t::Int64, β₁::Float64, β₂::Float64, ϵ::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, biases::Array{Float32}, Vdw::Array{Float32}, Sdw::Array{Float32}, Vdb::Array{Float32}, Sdb::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64, direction::Int64)
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            Vdw[a,b,c,d] *= β₁
            Sdw[a,b,c,d] *= β₂
        end
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        Vdw[x+1,y+1,j,k] += ∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]*(1-β₁)
                        Sdw[x+1,y+1,j,k] += (∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1])^2*(1-β₂)
                    end
                    index += input2D_size[2]
                end
            end
        end
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            filters[a,b,c,d] -= α*(Vdw[a,b,c,d]/(1-β₁^t))/(sqrt(Sdw[a,b,c,d]/(1-β₂^t))+ϵ)*direction
        end

        @avx for x in axes(filters, 1)
            Vdb[x] *= β₁
            Sdb[x] *= β₂
        end
        @avx for x in 0:size(filters, 1)-1, y in 1:unit_size[1]
            Vdb[x+1] += ∇biases[x*unit_size[1]+y, 1]*(1-β₁)/unit_size[1]
            Sdb[x+1] += ∇biases[x*unit_size[1]+y, 1]^2*(1-β₂)/unit_size[1]
        end
        @avx for x in axes(filters, 1)
            biases[x] -= α*(Vdb[x]/(1-β₁^t))/(sqrt(Sdb[x]/(1-β₂^t))+ϵ)*direction
        end
    end

    function AdaBelief!(α::Float64, t::Int64, β₁::Float64, β₂::Float64, ϵ::Float64, Last_Layer_output::Array{Float32}, ∇biases::Array{Float32}, biases::Array{Float32}, Vdw::Array{Float32}, Sdw::Array{Float32}, Vdb::Array{Float32}, Sdb::Array{Float32}, filters::Array{Float32}, step_x::Int64, step_y::Int64, input2D_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, unit_size::Tuple{Int64, Int64}, conv_num_per_row::Int64, conv_num_per_col::Int64, direction::Int64)
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            Vdw[a,b,c,d] *= β₁
            Sdw[a,b,c,d] *= β₂
            Sdw[a,b,c,d] += ϵ
        end
        @avx for x in 0:size(filters, 1)-1, y in 0:size(filters, 2)-1
            for i in 0:unit_size[1]-1
                index = step_x*(i%conv_num_per_row) + input2D_size[2]*step_y*(i÷conv_num_per_row)
                for j in 1:kernel_size[1]
                    for k in 1:kernel_size[2]
                        Vdw[x+1,y+1,j,k] += ∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]*(1-β₁)
                        Sdw[x+1,y+1,j,k] += (∇biases[x*unit_size[1]+i+1,1]*Last_Layer_output[y*unit_size[2]+index+k,1]-Vdw[x+1,y+1,j,k])^2*(1-β₂)
                    end
                    index += input2D_size[2]
                end
            end
        end
        @avx for a in axes(filters, 1), b in axes(filters, 2), c in axes(filters, 3), d in axes(filters, 4)
            filters[a,b,c,d] -= α*(Vdw[a,b,c,d]/(1-β₁^t))/(sqrt(Sdw[a,b,c,d]/(1-β₂^t))+ϵ)*direction
        end

        @avx for x in axes(filters, 1)
            Vdb[x] *= β₁
            Sdb[x] *= β₂
            Sdb[x] += ϵ
        end
        @avx for x in 0:size(filters, 1)-1, y in 1:unit_size[1]
            Vdb[x+1] += ∇biases[x*unit_size[1]+y, 1]*(1-β₁)/unit_size[1]
            Sdb[x+1] += (∇biases[x*unit_size[1]+y, 1]-Vdb[x+1])^2*(1-β₂)/unit_size[1]
        end
        @avx for x in axes(filters, 1)
            biases[x] -= α*(Vdb[x]/(1-β₁^t))/(sqrt(Sdb[x]/(1-β₂^t))+ϵ)*direction
        end
    end

    function save_Conv2D(layer::Conv2D, file::Any, id::Int64)
        write(file, string(id), "Conv2D")
        write(file, string(id)*"input_size", layer.input_size)
        write(file, string(id)*"filters", layer.filters)
        write(file, string(id)*"unit_size", collect(layer.unit_size))
        write(file, string(id)*"padding", layer.padding)
        write(file, string(id)*"input2D_size", collect(layer.input2D_size))
        write(file, string(id)*"step_x", layer.step_x)
        write(file, string(id)*"step_y", layer.step_y)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_Conv2D(layer::Conv2D, file::Any, id::Int64)
        layer.input_size = read(file, string(id)*"input_size")
        layer.filters = read(file, string(id)*"filters")
        layer.unit_size = Tuple(read(file, string(id)*"unit_size"))
        layer.kernel_size = (size(layer.filters,3), size(layer.filters,4))
        layer.padding = read(file, string(id)*"padding")
        layer.input2D_size = Tuple(read(file, string(id)*"input2D_size"))
        layer.step_x = read(file, string(id)*"step_x")
        layer.step_y = read(file, string(id)*"step_y")
        layer.Vdw = zeros(Float32, size(layer.filters))
        layer.Sdw = zeros(Float32, size(layer.filters))
        kernel_size = layer.kernel_size
        layer.conv_num_per_row = (layer.input2D_size[2]-kernel_size[2])÷layer.step_x+1
        layer.conv_num_per_col = (layer.input2D_size[1]-kernel_size[1])÷layer.step_y+1
        layer.layer_size = layer.conv_num_per_row*layer.conv_num_per_col*size(layer.filters, 1)
        layer.original_input2D_size = (layer.input2D_size[1]-2*layer.padding, layer.input2D_size[2]-2*layer.padding)
    end
end

# Source: https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
