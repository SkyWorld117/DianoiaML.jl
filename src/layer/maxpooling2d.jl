module MaxPooling2DM
    import ..DianoiaML: precision_type
    using ..DianoiaML: checked_precision
    import ..DianoiaML: backward!, supports_backward
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer
    mutable struct MaxPooling2D{T<:AbstractFloat} <: AbstractLayer
        input_shape::NTuple{3,Int}
        output_shape::NTuple{3,Int}
        pool_size::NTuple{2,Int}
        strides::NTuple{2,Int}

        index::Array{Int,4}
        output::Array{T,4}
        δ::Array{T,4}

        function MaxPooling2D(;input_shape::Tuple, pool_size::NTuple{2,Int}, strides::NTuple{2,Int}=pool_size, precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            length(input_shape) == 3 && all(>(0), input_shape) && all(>(0), strides) &&
                all(d -> 0 < pool_size[d] <= input_shape[d], 1:2) ||
                throw(ArgumentError("invalid pooling input dimensions, window or strides"))
            pool_num_per_col = (input_shape[1]-pool_size[1])÷strides[1]+1
            pool_num_per_row = (input_shape[2]-pool_size[2])÷strides[2]+1
            output_shape = (pool_num_per_col, pool_num_per_row, input_shape[3])
            new{T}(input_shape, output_shape, pool_size, strides)
        end
    end

    precision_type(::MaxPooling2D{T}) where {T} = T

    function initialize!(layer::MaxPooling2D{T}, mini_batch::Int) where {T}
        isdefined(layer, :output) && size(layer.output, 4) == mini_batch && return
        layer.index = zeros(Int, layer.output_shape..., mini_batch)
        layer.output = zeros(T, layer.output_shape..., mini_batch)
        layer.δ = zeros(T, layer.input_shape..., mini_batch)
    end

    function activate!(layer::MaxPooling2D{T}, input::Array{T,4}) where {T}
        size(input) == size(layer.δ) || throw(DimensionMismatch("unexpected pooling input size"))
        examples = axes(input, 4)
        # Each example writes to disjoint slices. Small batches avoid task overhead.
        if Threads.nthreads() > 1 && length(examples) > 1 && length(layer.output) >= 32_768
            Threads.@threads for example in examples
                pool_example!(layer, input, example)
            end
        else
            for example in examples
                pool_example!(layer, input, example)
            end
        end
        return nothing
    end

    function pool_example!(layer, input, example)
        sx, sy = layer.strides
        px, py = layer.pool_size
        linear = LinearIndices(input)
        width = size(input, 1)
        @inbounds for c in axes(layer.output, 3)
            for j in axes(layer.output, 2), i in axes(layer.output, 1)
                x, y = (i - 1) * sx + 1, (j - 1) * sy + 1
                first = linear[x, y, c, example]
                index = first
                value = input[first]
                for row in first:width:first + (py - 1) * width
                    for k in row:row + px - 1
                        pixel = input[k]
                        # Match findmax's ordering and retain the first index on ties.
                        if isless(value, pixel)
                            value = pixel
                            index = k
                        end
                    end
                end
                layer.output[i, j, c, example] = value
                layer.index[i, j, c, example] = index
            end
        end
    end

    supports_backward(::MaxPooling2D) = true

    function backward!(layer::MaxPooling2D{T}, input, δₗ₊₁) where {T}
        fill!(layer.δ, zero(T))
        # Overlapping windows can select the same input element.
        for i in eachindex(δₗ₊₁, layer.index)
            layer.δ[layer.index[i]] += δₗ₊₁[i]
        end
        return layer.δ
    end

    function update!(layer::MaxPooling2D{T}, optimizer::String, input, delta, α::Real, parameters::Tuple, direction::Int=1) where {T}
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
