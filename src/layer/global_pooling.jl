"""Average the two spatial axes, returning channels × batch."""
struct GlobalMeanPool2D <: AbstractLayer end

layer_cache(::GlobalMeanPool2D) = ActivationCache()
output_shape(::GlobalMeanPool2D, input) = (last(spatial_shape(input)),)

function forward!(::GlobalMeanPool2D, cache::ActivationCache, input::AbstractArray{T,4}, mode) where {T<:AbstractFloat}
    width, height, channels, batch = size(input)
    isempty(input) && throw(ArgumentError("GlobalMeanPool2D requires nonempty input"))
    A = accumulation_type(T)
    dims = (1, 1, channels, batch)
    if !isdefined(cache, :output) || size(cache.output) != dims ||
       eltype(cache.output) !== A || !same_device(cache.output, input)
        cache.output = similar(input, A, dims)
    end
    reduce_sum!(A, cache.output, input)
    broadcast_tensor!(/, cache.output, cache.output, A(width * height))
    result = reshape(cache.output, channels, batch)
    return A === T ? result : map_tensor(ConvertTo{T}(), result, T)
end

function backward!(::GlobalMeanPool2D, cache::ActivationCache, input, ∇y; compute_parameters=true)
    if !isdefined(cache, :gradient) || size(cache.gradient) != size(input) ||
       eltype(cache.gradient) !== eltype(input) || !same_device(cache.gradient, input)
        cache.gradient = similar(input)
    end
    width, height, channels, batch = size(input)
    A = accumulation_type(eltype(∇y))
    return broadcast_tensor!((dy, n) -> typeof(n)(dy) / n, cache.gradient,
                             reshape(∇y, 1, 1, channels, batch), A(width * height))
end
