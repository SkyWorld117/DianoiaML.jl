struct Dropout{T<:Real} <: AbstractLayer
    rate::T
    function Dropout(rate::T) where {T<:Real}
        0 <= rate < 1 || throw(ArgumentError("dropout rate must be in [0, 1)"))
        new{T}(rate)
    end
end

output_shape(::Dropout, input) = input

mutable struct DropoutCache
    active::Bool
    samples::AbstractArray
    output::AbstractArray
    gradient::AbstractArray
    DropoutCache() = new(false)
end

layer_cache(::Dropout) = DropoutCache()

function forward!(layer::Dropout, cache::DropoutCache, input, mode)
    cache.active = mode isa TrainingMode && !iszero(layer.rate)
    cache.active || return input
    if !isdefined(cache, :output) || size(cache.output) != size(input) ||
       eltype(cache.output) !== eltype(input) || !same_device(cache.output, input)
        cache.output, cache.gradient = similar(input), similar(input)
        A = accumulation_type(eltype(input))
        cache.samples = similar(input, random_precision(A))
    end
    if gpu_array(input)
        launch!(random_uniform_gpu!, cache.samples, rand(mode.rng, UInt64))
    else
        rand!(mode.rng, cache.samples)
    end
    return apply_dropout!(cache.output, layer, cache.samples, input)
end

function apply_dropout!(dst, layer, samples, input)
    A = accumulation_type(eltype(input))
    p = A(layer.rate)
    scale = inv(one(A) - p)
    return dst .= ifelse.(samples .≥ p, input .* scale, zero(A))
end

function backward!(layer::Dropout, cache::DropoutCache, input, ∇y; compute_parameters=true)
    cache.active || return ∇y
    return apply_dropout!(cache.gradient, layer, cache.samples, ∇y)
end
