module MaxPooling2DM
    import ..DianoiaML: backward!, supports_backward
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer
    mutable struct MaxPooling2D <: AbstractLayer
        input_shape::NTuple{3,Int}
        output_shape::NTuple{3,Int}
        pool_size::NTuple{2,Int}
        strides::NTuple{2,Int}

        index::Array{Int,4}
        output::Array{Float32,4}
        δ::Array{Float32,4}

        function MaxPooling2D(;input_shape::Tuple, pool_size::NTuple{2,Int}, strides::NTuple{2,Int}=pool_size)
            length(input_shape) == 3 && all(>(0), input_shape) && all(>(0), strides) &&
                all(d -> 0 < pool_size[d] <= input_shape[d], 1:2) ||
                throw(ArgumentError("invalid pooling input dimensions, window or strides"))
            pool_num_per_col = (input_shape[1]-pool_size[1])÷strides[1]+1
            pool_num_per_row = (input_shape[2]-pool_size[2])÷strides[2]+1
            output_shape = (pool_num_per_col, pool_num_per_row, input_shape[3])
            new(input_shape, output_shape, pool_size, strides)
        end
    end

    function initialize!(layer::MaxPooling2D, mini_batch::Int)
        isdefined(layer, :output) && size(layer.output, 4) == mini_batch && return
        layer.index = zeros(Int, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate!(layer::MaxPooling2D, input::Array{Float32,4})
        size(input) == size(layer.δ) || throw(DimensionMismatch("unexpected pooling input size"))
        sx, sy = layer.strides
        px, py = layer.pool_size
        linear = LinearIndices(input)
        @inbounds for b in axes(layer.output, 4), c in axes(layer.output, 3)
            for j in axes(layer.output, 2), i in axes(layer.output, 1)
                x, y = (i - 1) * sx, (j - 1) * sy
                value, index = findmax(view(input, x + 1:x + px, y + 1:y + py, c, b))
                layer.output[i, j, c, b] = value
                layer.index[i, j, c, b] = linear[x + index[1], y + index[2], c, b]
            end
        end
    end

    supports_backward(::MaxPooling2D) = true

    function backward!(layer::MaxPooling2D, input, δₗ₊₁)
        fill!(layer.δ, 0f0)
        # Overlapping windows can select the same input element.
        for i in eachindex(δₗ₊₁, layer.index)
            layer.δ[layer.index[i]] += δₗ₊₁[i]
        end
        return layer.δ
    end

    function update!(layer::MaxPooling2D, optimizer::String, input, delta, α::Float64, parameters::Tuple, direction::Int=1)
        backward!(layer, input, delta)
    end

    function save_layer(layer::MaxPooling2D, file, id::String)
        write(file, id, "MaxPooling2D")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"pool_size", collect(layer.pool_size))
        write(file, id*"strides", collect(layer.strides))
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        pool_size = tuple(read(file, id*"pool_size")...)
        strides = tuple(read(file, id*"strides")...)
        return (input_shape=input_shape, pool_size=pool_size, strides=strides)
    end
end
