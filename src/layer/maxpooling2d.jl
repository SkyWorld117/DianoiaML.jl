struct MaxPooling2D <: AbstractLayer
    window::NTuple{2,Int}
    stride::NTuple{2,Int}
    function MaxPooling2D(window::NTuple{2,Int}; stride::NTuple{2,Int}=window)
        all(>(0), window) && all(>(0), stride) || throw(ArgumentError("pool window and stride must be positive"))
        new(window, stride)
    end
end

function output_shape(layer::MaxPooling2D, input)
    dims = spatial_shape(input)
    spatial = ntuple(2) do d
        dims[d] === nothing && return nothing
        dims[d] >= layer.window[d] || throw(DimensionMismatch("pool window exceeds input"))
        return fld(dims[d] - layer.window[d], layer.stride[d]) + 1
    end
    return (spatial..., dims[3])
end

mutable struct PoolCache
    indices::Array{Int,4}
    output::Array{<:AbstractFloat,4}
    input_gradient::Array{<:AbstractFloat,4}
    PoolCache() = new()
end

layer_cache(::MaxPooling2D) = PoolCache()

function forward!(layer::MaxPooling2D, cache::PoolCache, input::Array{T,4}, mode) where {T<:AbstractFloat}
    if !isdefined(cache, :input_gradient) || size(cache.input_gradient) != size(input) || eltype(cache.input_gradient) !== T
        dims = (shape(layer, size(input)[1:3])..., size(input, 4))
        cache.indices = Array{Int}(undef, dims)
        cache.output = Array{T}(undef, dims)
        cache.input_gradient = similar(input)
    end
    return pool!(layer, input, cache.output, cache.indices)
end

function pool!(layer, input, output, indices)
    examples = axes(input, 4)
    # Pool slices for different examples can be written independently.
    if Threads.nthreads() > 1 && length(examples) > 1 && length(output) >= 32_768
        Threads.@threads for example in examples
            pool_example!(layer, input, output, indices, example)
        end
    else
        for example in examples
            pool_example!(layer, input, output, indices, example)
        end
    end
    return output
end

function pool_example!(layer, input, output, indices, example)
    sx, sy = layer.stride
    px, py = layer.window
    linear = LinearIndices(input)
    width = size(input, 1)
    @inbounds for c in axes(output, 3), j in axes(output, 2), i in axes(output, 1)
        x, y = (i - 1) * sx + 1, (j - 1) * sy + 1
        first = linear[x, y, c, example]
        index = first
        value = input[first]
        for row in first:width:first + (py - 1) * width, k in row:row + px - 1
            pixel = input[k]
            if isless(value, pixel)
                value, index = pixel, k
            end
        end
        output[i, j, c, example] = value
        indices[i, j, c, example] = index
    end
end

backward!(::MaxPooling2D, cache::PoolCache, input, ∇y; compute_parameters=true) =
    pool_gradient!(cache.input_gradient, ∇y, cache.indices)

function pool_gradient!(∇x::Array{T,4}, ∇y, indices) where {T}
    fill!(∇x, zero(T))
    # Overlapping windows can select the same element.
    @inbounds for i in eachindex(∇y, indices)
        ∇x[indices[i]] += ∇y[i]
    end
    return ∇x
end
