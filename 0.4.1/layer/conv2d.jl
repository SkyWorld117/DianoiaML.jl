module Conv2DM
    include("../tools/glorotuniform.jl")
    using .GlorotUniform
    using LoopVectorization

    mutable struct Conv2D
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple
        kernel_size::Tuple{Int64, Int64}
        activation_function::Module

        filters::Array{Float32}
        biases::Array{Float32}

        padding::Int64
        filter::Int64
        strides::Tuple{Int64, Int64}

        ∇weights::Array{Float32}
        Vdw::Array{Float32}
        Sdw::Array{Float32}
        ∇biases_c::Array{Float32}
        Vdb::Array{Float32}
        Sdb::Array{Float32}

        padding_input::Array{Float32}
        value::Array{Float32}
        output::Array{Float32}

        ∇biases::Array{Float32}
        pre_δ::Array{Float32}
        δ::Array{Float32}

        function Conv2D(;input_shape::Tuple, filter::Int64, padding::Int64=0, kernel_size::Tuple{Int64, Int64}, strides::Tuple{Int64, Int64}=(1,1), activation_function::Module, randomization::Bool=true)
            conv_num_per_col = (input_shape[1]+2*padding-kernel_size[1])÷strides[1]+1
            conv_num_per_row = (input_shape[2]+2*padding-kernel_size[2])÷strides[2]+1
            output_shape = (conv_num_per_col, conv_num_per_row, filter)

            input_size = 1
            for i in input_shape
                input_size *= i
            end

            filter_size = (filter, input_shape[3], kernel_size[1], kernel_size[2])
            filters = !randomization ? zeros(Float32, filter_size) : Array{Float32}(GUMatrix(input_size, conv_num_per_row*conv_num_per_col*filter, filter_size))
            biases = !randomization ? zeros(Float32, filter) : Array{Float32}(GUMatrix(input_size, conv_num_per_row*conv_num_per_col*filter, filter))

            ∇weights = zeros(Float32, filter_size)
            Vdw = zeros(Float32, filter_size)
            Sdw = zeros(Float32, filter_size)
            ∇biases_c = zeros(Float32, filter)
            Vdb = zeros(Float32, filter)
            Sdb = zeros(Float32, filter)
            new(save_Conv2D, load_Conv2D, activate_Conv2D, init_Conv2D, update_Conv2D, input_shape, output_shape, kernel_size, activation_function, filters, biases, padding, filter, strides, ∇weights, Vdw, Sdw, ∇biases_c, Vdb, Sdb)
        end
    end

    function init_Conv2D(layer::Conv2D, mini_batch::Int64)
        layer.padding_input = zeros(Float32, (layer.input_shape[1]+2*layer.padding, layer.input_shape[2]+2*layer.padding, layer.input_shape[3], mini_batch))
        layer.value = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)

        layer.∇biases = zeros(Float32, layer.output_shape..., mini_batch)
        layer.pre_δ = zeros(Float32, (layer.input_shape[1]+2*layer.padding, layer.input_shape[2]+2*layer.padding, layer.input_shape[3], mini_batch))
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate_Conv2D(layer::Conv2D, input::Array{Float32})
        @avxt for i in axes(input, 1), j in axes(input, 2), c in axes(input, 3), b in axes(input, 4)
            layer.padding_input[i+layer.padding, j+layer.padding, c, b] = input[i, j, c, b]
        end

        x, y = layer.strides
        @avxt for i in axes(layer.value, 1), j in axes(layer.value, 2), f in axes(layer.value, 3), b in axes(layer.value, 4)
            s = 0.0f0
            for k₁ in axes(layer.filters, 3), k₂ in axes(layer.filters, 4), c in axes(layer.filters, 2)
                s += layer.padding_input[(i-1)*x+k₁, (j-1)*y+k₂, c, b] * layer.filters[f, c, k₁, k₂]
            end
            layer.value[i, j, f, b] = s + layer.biases[f]
        end

        layer.activation_function.func!(layer.output, layer.value)
    end

    function update_Conv2D(layer::Conv2D, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        x, y = layer.strides
        layer.activation_function.get_∇biases!(layer.∇biases, layer.value, δₗ₊₁)
        @avxt layer.pre_δ .= 0.0f0

        @avxt for i in axes(layer.∇biases, 1), j in axes(layer.∇biases, 2), f in axes(layer.∇biases, 3), b in axes(layer.∇biases, 4)
            for k₁ in axes(layer.filters, 3), k₂ in axes(layer.filters, 4), c in axes(layer.filters, 2)
                layer.pre_δ[(i-1)*x+k₁, (j-1)*y+k₂, c, b] += layer.filters[f, c, k₁, k₂]*layer.∇biases[i, j, f, b]
            end
        end

        @avxt for i in axes(layer.δ, 1), j in axes(layer.δ, 2), c in axes(layer.δ, 3), b in axes(layer.δ, 4)
            layer.δ[i, j, c, b] = layer.pre_δ[layer.padding+i, layer.padding+j, c, b]
        end

        if optimizer=="SGD"
            @avxt for i in axes(layer.∇biases, 1), j in axes(layer.∇biases, 2), f in axes(layer.∇biases, 3)
                for k₁ in axes(layer.filters, 3), k₂ in axes(layer.filters, 4), c in axes(layer.filters, 2)
                    layer.filters[f, c, k₁, k₂] -= α*layer.padding_input[(i-1)*x+k₁, (j-1)*y+k₂, c, 1]*layer.∇biases[i, j, f, 1]*direction
                end
                layer.biases[f] -= α*layer.∇biases[i, j, f, 1]*direction
            end

        elseif optimizer=="Minibatch_GD"
            batch_size = size(layer.∇biases, 4)
            @avxt for i in axes(layer.∇biases, 1), j in axes(layer.∇biases, 2), f in axes(layer.∇biases, 3), b in axes(layer.∇biases, 4)
                for k₁ in axes(layer.filters, 3), k₂ in axes(layer.filters, 4), c in axes(layer.filters, 2)
                    layer.filters[f, c, k₁, k₂] -= α*layer.padding_input[(i-1)*x+k₁, (j-1)*y+k₂, c, b]*layer.∇biases[i, j, f, b]/batch_size*direction
                end
                layer.biases[f] -= α*layer.∇biases[i, j, f, b]/batch_size*direction
            end

        elseif optimizer=="Adam"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]

            @avxt layer.∇weights .= 0.0f0
            @avxt layer.∇biases_c .= 0.0f0

            @avxt for i in axes(layer.∇biases, 1), j in axes(layer.∇biases, 2), f in axes(layer.∇biases, 3)
                for k₁ in axes(layer.filters, 3), k₂ in axes(layer.filters, 4), c in axes(layer.filters, 2)
                    layer.∇weights[f, c, k₁, k₂] += layer.padding_input[(i-1)*x+k₁, (j-1)*y+k₂, c, 1]*layer.∇biases[i, j, f, 1]
                end
                layer.∇biases_c[f] += layer.∇biases[i, j, f, 1]
            end

            @avxt for i in eachindex(layer.filters)
                layer.Vdw[i] = β₁*layer.Vdw[i] + (1-β₁)*layer.∇weights[i]
                layer.Sdw[i] = β₂*layer.Sdw[i] + (1-β₂)*layer.∇weights[i]^2
                layer.filters[i] -= α*(layer.Vdw[i]/(1-β₁^t))/(sqrt(layer.Sdw[i]/(1-β₂^t))+ϵ)*direction
            end
            @avxt for i in eachindex(layer.biases)
                layer.Vdb[i] = β₁*layer.Vdb[i] + (1-β₁)*layer.∇biases_c[i]
                layer.Sdb[i] = β₂*layer.Sdb[i] + (1-β₂)*layer.∇biases_c[i]^2
                layer.biases[i] -= α*(layer.Vdb[i]/(1-β₁^t))/(sqrt(layer.Sdb[i]/(1-β₂^t))+ϵ)*direction
            end

        elseif optimizer=="AdaBelief"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]

            @avxt layer.∇weights .= 0.0f0
            @avxt layer.∇biases_c .= 0.0f0

            @avxt for i in axes(layer.∇biases, 1), j in axes(layer.∇biases, 2), f in axes(layer.∇biases, 3)
                for k₁ in axes(layer.filters, 3), k₂ in axes(layer.filters, 4), c in axes(layer.filters, 2)
                    layer.∇weights[f, c, k₁, k₂] += layer.padding_input[(i-1)*x+k₁, (j-1)*y+k₂, c, 1]*layer.∇biases[i, j, f, 1]
                end
                layer.∇biases_c[f] += layer.∇biases[i, j, f, 1]
            end

            @avxt for i in eachindex(layer.filters)
                layer.Vdw[i] = β₁*layer.Vdw[i] + (1-β₁)*layer.∇weights[i]
                layer.Sdw[i] = β₂*layer.Sdw[i] + (1-β₂)*(layer.∇weights[i]-layer.Vdw[i])^2
                layer.filters[i] -= α*(layer.Vdw[i]/(1-β₁^t))/(sqrt(layer.Sdw[i]/(1-β₂^t))+ϵ)*direction
            end
            @avxt for i in eachindex(layer.biases)
                layer.Vdb[i] = β₁*layer.Vdb[i] + (1-β₁)*layer.∇biases_c[i]
                layer.Sdb[i] = β₂*layer.Sdb[i] + (1-β₂)*(layer.∇biases_c[i]-layer.Vdb[i])^2
                layer.biases[i] -= α*(layer.Vdb[i]/(1-β₁^t))/(sqrt(layer.Sdb[i]/(1-β₂^t))+ϵ)*direction
            end
        end
    end

    function save_Conv2D(layer::Conv2D, file::Any, id::Int64)
        write(file, string(id), "Conv2D")
        write(file, string(id)*"input_shape", collect(layer.input_shape))
        write(file, string(id)*"filter", layer.filter)
        write(file, string(id)*"padding", layer.padding)
        write(file, string(id)*"kernel_size", collect(layer.kernel_size))
        write(file, string(id)*"strides", collect(layer.strides))
        write(file, string(id)*"activation_function", layer.activation_function.get_name())

        write(file, string(id)*"filters", layer.filters)
        write(file, string(id)*"biases", layer.biases)
    end

    function load_Conv2D(layer::Conv2D, file::Any, id::Int64)
        layer.filters = read(file, string(id)*"filters")
        layer.biases = read(file, string(id)*"biases")
    end

    function get_args(file::Any, id::Int64)
        input_shape = tuple(read(file, string(id)*"input_shape")...)
        filter = read(file, string(id)*"filter")
        padding = read(file, string(id)*"padding")
        kernel_size = tuple(read(file, string(id)*"kernel_size")...)
        strides = tuple(read(file, string(id)*"strides")...)
        return (input_shape=input_shape, filter=filter, padding=padding, kernel_size=kernel_size, strides=strides)
    end
end
