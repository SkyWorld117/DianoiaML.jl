module MaxPooling2DM
    using LoopVectorization, Polyester

    mutable struct MaxPooling2D
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple
        pool_size::Tuple{Int64, Int64}
        strides::Tuple{Int64, Int64}

        index::Array{Int64}
        output::Array{Float32}
        δ::Array{Float32}

        function MaxPooling2D(;input_shape::Tuple, pool_size::Tuple{Int64, Int64}, strides::Tuple{Int64,Int64}=(pool_size[1],pool_size[2]))
            pool_num_per_col = (input_shape[1]-pool_size[1])÷strides[1]+1
            pool_num_per_row = (input_shape[2]-pool_size[2])÷strides[2]+1
            output_shape = (pool_num_per_col, pool_num_per_row, input_shape[3])
            new(save_MaxPooling2D, load_MaxPooling2D, activate_MaxPooling2D, init_MaxPooling2D, update_MaxPooling2D, input_shape, output_shape, pool_size, strides)
        end
    end

    function init_MaxPooling2D(layer::MaxPooling2D, mini_batch::Int64)
        layer.index = zeros(Int64, 2, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate_MaxPooling2D(layer::MaxPooling2D, input::Array{Float32})
        x, y = layer.strides
        # Waiting for the update of LoopVectorization
        #=@avxt for i in axes(layer.output, 1), j in axes(layer.output, 2), c in axes(layer.output, 3), b in axes(layer.output, 4)
            s = -Inf32
            for p₁ in 1:layer.pool_size[1], p₂ in 1:layer.pool_size[2]
                s = ifelse(input[(i-1)*x+p₁, (j-1)*y+p₂, c, b]>=layer.output[i, j, c, b], input[(i-1)*x+p₁, (j-1)*y+p₂, c, b], s)
                layer.index[1, i, j, c, b] = ifelse(input[(i-1)*x+p₁, (j-1)*y+p₂, c, b]>=s, (i-1)*x+p₁, layer.index[1, i, j, c, b])
                layer.index[2, i, j, c, b] = ifelse(input[(i-1)*x+p₁, (j-1)*y+p₂, c, b]>=s, (j-1)*y+p₂, layer.index[2, i, j, c, b])
            end
            layer.output[i, j, c, b] = s
        end=#

        @batch for i in axes(layer.output, 1), j in axes(layer.output, 2), c in axes(layer.output, 3), b in axes(layer.output, 4)
            temp = findmax(view(input, (i-1)*x+1:(i-1)*x+layer.pool_size[1], (j-1)*y+1:(j-1)*y+layer.pool_size[2], c, b))
            layer.output[i, j, c, b] = temp[1]
            layer.index[1, i, j, c, b] = (i-1)*x+temp[2][1]
            layer.index[2, i, j, c, b] = (j-1)*y+temp[2][2]
        end
    end

    function update_MaxPooling2D(layer::MaxPooling2D, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=0)
        @avxt layer.δ .= 0.0f0
        @batch for i in axes(δₗ₊₁, 1), j in axes(δₗ₊₁, 2), c in axes(δₗ₊₁, 3), b in axes(δₗ₊₁, 4)
            layer.δ[layer.index[1, i, j, c, b], layer.index[2, i, j, c, b], c, b] += δₗ₊₁[i, j, c, b]
        end
    end

    function save_MaxPooling2D(layer::MaxPooling2D, file::Any, id::String)
        write(file, id, "MaxPooling2D")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"pool_size", collect(layer.pool_size))
        write(file, id*"strides", collect(layer.strides))
    end

    function load_MaxPooling2D(layer::MaxPooling2D, file::Any, id::String)
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        pool_size = tuple(read(file, id*"pool_size")...)
        strides = tuple(read(file, id*"strides")...)
        return (input_shape=input_shape, pool_size=pool_size, strides=strides)
    end
end
