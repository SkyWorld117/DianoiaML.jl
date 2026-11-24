"""A learned projection on the first axis of a tensor."""
struct Dense{T<:AbstractFloat,F<:AbstractActivation} <: AbstractLayer
    weights::Matrix{T}
    bias::Vector{T}
    activation::F
end

function Dense(input::Int, output::Int; activation::AbstractActivation=Identity(),
               randomization::Bool=true, precision::Type{T}=Float32) where {T<:AbstractFloat}
    checked_precision(T)
    input > 0 && output > 0 || throw(ArgumentError("Dense features must be positive"))
    weights = randomization ? GUMatrix(input, output, (output, input), T) : zeros(T, output, input)
    return Dense(weights, zeros(T, output), activation)
end

precision_type(::Dense{T}) where {T} = T
input_shape(layer::Dense) = (size(layer.weights, 2), Colon())
output_shape(layer::Dense, input) = feature_shape(input, size(layer.weights, 1))
layer_parameters(layer::Dense) = (; weights=layer.weights, bias=layer.bias)

mutable struct DenseCache{T,A}
    weights::Matrix{A}
    bias::Vector{A}
    values::Matrix{T}
    delta::Matrix{T}
    output::Array{T}
    input_gradient::Array{T}
    DenseCache{T,A}(weights, bias) where {T,A} = new{T,A}(weights, bias)
end

function layer_cache(layer::Dense{T}) where {T}
    A = accumulation_type(T)
    return DenseCache{T,A}(similar(layer.weights, A), similar(layer.bias, A))
end
layer_gradients(::Dense, cache::DenseCache) = (; weights=cache.weights, bias=cache.bias)

function multiply!(dest, left, right, ::Type{A}, scale=one(A)) where {A}
    x = eltype(left) === A ? left : A.(left)
    y = eltype(right) === A ? right : A.(right)
    result = eltype(dest) === A ? dest : similar(dest, A)
    mul!(result, x, y, scale, zero(A))
    result === dest || copyto!(dest, result)
    return dest
end

function forward!(layer::Dense{T}, cache::DenseCache{T,A}, input::Array{T}, mode) where {T,A}
    ndims(input) >= 2 && size(input, 1) == size(layer.weights, 2) ||
        throw(DimensionMismatch("Dense expects $(size(layer.weights, 2)) features on the first axis"))
    if !isdefined(cache, :input_gradient) || size(cache.input_gradient) != size(input)
        dims = (size(layer.weights, 1), Base.tail(size(input))...)
        cache.values = Matrix{T}(undef, first(dims), prod(Base.tail(dims)))
        cache.delta = similar(cache.values)
        cache.output = Array{T}(undef, dims)
        cache.input_gradient = similar(input)
    end
    multiply!(cache.values, layer.weights, reshape(input, size(input, 1), :), A)
    cache.values .+= layer.bias
    activate_values!(reshape(cache.output, size(cache.values)), layer.activation, cache.values)
    return cache.output
end

function backward!(layer::Dense{T}, cache::DenseCache{T,A}, input, delta; compute_parameters=true) where {T,A}
    size(delta) == size(cache.output) || throw(DimensionMismatch("Dense cotangent shape differs"))
    activation_gradient!(cache.delta, layer.activation, cache.values, reshape(cache.output, size(cache.values)),
                         reshape(delta, size(cache.values)))
    width = size(input, 1)
    multiply!(reshape(cache.input_gradient, width, :), transpose(layer.weights), cache.delta, A)
    if compute_parameters
        batch = A(size(input, ndims(input)))
        multiply!(cache.weights, cache.delta, transpose(reshape(input, width, :)), A, inv(batch))
        fill!(cache.bias, zero(A))
        for b in axes(cache.delta, 2)
            @inbounds @simd for i in axes(cache.delta, 1)
                cache.bias[i] += convert(A, cache.delta[i, b])
            end
        end
        cache.bias ./= batch
    end
    return cache.input_gradient
end
