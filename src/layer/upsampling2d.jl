struct UpSampling2D{F<:AbstractActivation} <: AbstractLayer
    scale::NTuple{2,Int}
    activation::F
    function UpSampling2D(scale::NTuple{2,Int}; activation::F=Identity()) where {F<:AbstractActivation}
        all(>(0), scale) || throw(ArgumentError("upsampling scale must be positive"))
        new{F}(scale, activation)
    end
end

function output_shape(layer::UpSampling2D, input)
    x, y, channels = spatial_shape(input)
    return (x === nothing ? nothing : x * layer.scale[1],
            y === nothing ? nothing : y * layer.scale[2], channels)
end

mutable struct UpsamplingCache
    values::Array
    output::Array
    delta::Array
    input_gradient::Array
    UpsamplingCache() = new()
end

layer_cache(::UpSampling2D) = UpsamplingCache()

function forward!(layer::UpSampling2D, cache::UpsamplingCache, input::Array{T,4}, mode) where {T<:AbstractFloat}
    dims = (shape(layer, size(input)[1:3])..., size(input, 4))
    if !isdefined(cache, :values) || size(cache.values) != dims || eltype(cache.values) !== T
        cache.values = Array{T}(undef, dims)
        cache.output, cache.delta = similar(cache.values), similar(cache.values)
        cache.input_gradient = similar(input)
    end
    upsample!(cache.values, input, layer.scale)
    return activate_values!(cache.output, layer.activation, cache.values)
end

function upsample!(output, input, scale)
    x, y = scale
    @inbounds for b in axes(input, 4), c in axes(input, 3), j in axes(input, 2), sy in 1:y, i in axes(input, 1)
        @simd for sx in 1:x
            output[(i - 1) * x + sx, (j - 1) * y + sy, c, b] = input[i, j, c, b]
        end
    end
end

function backward!(layer::UpSampling2D, cache::UpsamplingCache, input, ∇y; compute_parameters=true)
    activation_gradient!(cache.delta, layer.activation, cache.values, cache.output, ∇y)
    return upsample_gradient!(cache.input_gradient, cache.delta, layer.scale)
end

function upsample_gradient!(∇x, δ, scale)
    A = accumulation_type(eltype(δ))
    x, y = scale
    @inbounds for b in axes(∇x, 4), c in axes(∇x, 3), j in axes(∇x, 2), i in axes(∇x, 1)
        total = zero(A)
        for sy in 1:y
            @simd for sx in 1:x
                total += convert(A, δ[(i - 1) * x + sx, (j - 1) * y + sy, c, b])
            end
        end
        ∇x[i, j, c, b] = total
    end
    return ∇x
end
