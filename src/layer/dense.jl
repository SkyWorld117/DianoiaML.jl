"""A learned projection on the first axis of a tensor."""
struct Dense{T<:AbstractFloat,F<:AbstractActivation,W<:AbstractMatrix{T},B<:AbstractVector{T}} <: AbstractLayer
    weights::W
    bias::B
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

mutable struct DenseCache{T,A,W<:AbstractMatrix{A},B<:AbstractVector{A},V<:AbstractMatrix{T}}
    weights::W
    bias::B
    values::V
    delta::V
    output::AbstractArray{T}
    input_gradient::AbstractArray{T}
    DenseCache(w::W, b::B, ::Type{V}) where {T,A,W<:AbstractMatrix{A},B<:AbstractVector{A},V<:AbstractMatrix{T}} =
        new{T,A,W,B,V}(w, b)
end

function layer_cache(layer::Dense{T}) where {T}
    A = accumulation_type(T)
    return DenseCache(similar(layer.weights, A), similar(layer.bias, A), typeof(layer.weights))
end
layer_gradients(::Dense, cache::DenseCache) = (; weights=cache.weights, bias=cache.bias)

function multiply!(dest, left, right, ::Type{A}, α=one(A)) where {A}
    gpu_array(dest) && return gpu_multiply!(dest, left, right, A, α)
    x = eltype(left) === A ? left : A.(left)
    y = eltype(right) === A ? right : A.(right)
    result = eltype(dest) === A ? dest : similar(dest, A)
    mul!(result, x, y, α, zero(A))
    result === dest || copyto!(dest, result)
    return dest
end

function forward!(layer::Dense{T}, cache::DenseCache{T,A}, input::AbstractArray{T}, mode) where {T,A}
    ndims(input) >= 2 && size(input, 1) == size(layer.weights, 2) ||
        throw(DimensionMismatch("Dense expects $(size(layer.weights, 2)) features on the first axis"))
    if !isdefined(cache, :input_gradient) || size(cache.input_gradient) != size(input)
        dims = (size(layer.weights, 1), Base.tail(size(input))...)
        cache.values = similar(layer.weights, T, (first(dims), prod(Base.tail(dims))))
        cache.delta = similar(cache.values)
        cache.output = similar(input, T, dims)
        cache.input_gradient = similar(input)
    end
    z = cache.values
    multiply!(z, layer.weights, reshape(input, size(input, 1), :), A)
    output = reshape(cache.output, size(z))
    if gpu_array(z) && layer.activation isa PointwiseActivation
        launch!(dense_epilogue_gpu!, output, z, layer.bias, layer.activation,
                pointwise_arguments(layer.activation, T))
    else
        z .+= layer.bias
        activate_values!(output, layer.activation, z)
    end
    return cache.output
end

function backward!(layer::Dense{T}, cache::DenseCache{T,A}, input, ∇y; compute_parameters=true) where {T,A}
    size(∇y) == size(cache.output) || throw(DimensionMismatch("Dense cotangent shape differs"))
    z, δ = cache.values, cache.delta
    activation_gradient!(δ, layer.activation, z, reshape(cache.output, size(z)), reshape(∇y, size(z)))
    width = size(input, 1)
    ∇x = reshape(cache.input_gradient, width, :)
    multiply!(∇x, transpose(layer.weights), δ, A)
    if compute_parameters
        batch = A(size(input, ndims(input)))
        ∇W, ∇b = cache.weights, cache.bias
        multiply!(∇W, δ, transpose(reshape(input, width, :)), A, inv(batch))
        if gpu_array(∇b)
            column_sum!(∇b, δ, batch)
            return cache.input_gradient
        end
        fill!(∇b, zero(A))
        for b in axes(δ, 2)
            @inbounds @simd for i in axes(δ, 1)
                ∇b[i] += convert(A, δ[i, b])
            end
        end
        ∇b ./= batch
    end
    return cache.input_gradient
end
