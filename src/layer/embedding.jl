using NNlib: gather!, scatter!

"""Look up learned vectors for one-based integer IDs with examples on the last axis."""
struct Embedding{T<:AbstractFloat,W<:AbstractMatrix{T}} <: AbstractLayer
    weights::W
end

function Embedding(vocabulary::Int, width::Int; precision::Type{T}=Float32) where {T<:AbstractFloat}
    checked_precision(T)
    vocabulary > 0 && width > 0 || throw(ArgumentError("Embedding dimensions must be positive"))
    return Embedding(GUMatrix(vocabulary, width, (width, vocabulary), T))
end

layer_parameters(layer::Embedding) = (; weights=layer.weights)
output_shape(layer::Embedding, input) =
    input === nothing ? (size(layer.weights, 1), Colon()) : (size(layer.weights, 1), input...)

mutable struct EmbeddingCache{W}
    gradient::W
    output::AbstractArray
    EmbeddingCache(gradient::W) where {W} = new{W}(gradient)
end

layer_cache(layer::Embedding{T}) where {T} = EmbeddingCache(similar(layer.weights, accumulation_type(T)))
layer_gradients(::Embedding, cache::EmbeddingCache) = (; weights=cache.gradient)

function forward!(layer::Embedding, cache::EmbeddingCache, ids::AbstractArray{<:Integer}, mode)
    ndims(ids) >= 1 && !isempty(ids) || throw(ArgumentError("Embedding requires nonempty batched IDs"))
    dims = (size(layer.weights, 1), size(ids)...)
    if !isdefined(cache, :output) || size(cache.output) != dims
        cache.output = similar(layer.weights, dims)
    end
    return gather!(cache.output, layer.weights, ids)
end

function backward!(::Embedding, cache::EmbeddingCache, ids, ∇y; compute_parameters=true)
    if compute_parameters
        fill!(cache.gradient, zero(eltype(cache.gradient)))
        scatter!(+, cache.gradient, ∇y, ids; checkbounds=false)
        cache.gradient ./= eltype(cache.gradient)(batch_examples(ids))
    end
    return nothing
end

"""Add learned position vectors to a features × tokens × batch tensor."""
struct PositionalEmbedding{T<:AbstractFloat,W<:AbstractMatrix{T}} <: AbstractLayer
    weights::W
end

function PositionalEmbedding(width::Int, context_length::Int; precision::Type{T}=Float32) where {T<:AbstractFloat}
    checked_precision(T)
    width > 0 && context_length > 0 || throw(ArgumentError("position embedding dimensions must be positive"))
    return PositionalEmbedding(GUMatrix(width, context_length, (width, context_length), T))
end

precision_type(::PositionalEmbedding{T}) where {T} = T
input_shape(layer::PositionalEmbedding) = (size(layer.weights, 1), nothing)

function output_shape(layer::PositionalEmbedding, input)
    input === nothing && return input_shape(layer)
    tokens = input[2]
    if tokens isa Int
        0 < tokens <= size(layer.weights, 2) || throw(DimensionMismatch("sequence exceeds the context length"))
    end
    return input
end

layer_parameters(layer::PositionalEmbedding) = (; weights=layer.weights)
layer_cache(layer::PositionalEmbedding{T}) where {T} =
    EmbeddingCache(similar(layer.weights, accumulation_type(T)))
layer_gradients(::PositionalEmbedding, cache::EmbeddingCache) = (; weights=cache.gradient)

function forward!(layer::PositionalEmbedding{T}, cache::EmbeddingCache, input::AbstractArray{T,3}, mode) where {T}
    shape(layer, size(input)[1:2])
    if !isdefined(cache, :output) || size(cache.output) != size(input)
        cache.output = similar(input)
    end
    positions = view(layer.weights, :, 1:size(input, 2))
    return cache.output .= input .+ positions
end

function backward!(::PositionalEmbedding, cache::EmbeddingCache, input, ∇y; compute_parameters=true)
    if compute_parameters
        A = eltype(cache.gradient)
        fill!(cache.gradient, zero(A))
        tokens = size(input, 2)
        positions = view(cache.gradient, :, 1:tokens)
        reduce_sum!(A, positions, ∇y)
        positions ./= A(size(input, 3))
    end
    return ∇y
end
