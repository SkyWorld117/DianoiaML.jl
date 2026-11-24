struct AttentionWorkspace{T,A}
    probabilities::Array{A,4}
    score_gradient::Matrix{A}
    query::Union{Array{A,3},Nothing}
    key::Union{Array{A,3},Nothing}
    value::Union{Array{A,3},Nothing}
    attended::Array{A,3}
    projection_input::Array{T,3}
    query_gradient::Array{A,3}
    key_gradient::Array{A,3}
    value_gradient::Array{A,3}
end

function AttentionWorkspace(::Type{T}, ::Type{A}, query_shape, key_shape, heads) where {T,A}
    queries, keys, batch = query_shape[2], key_shape[2], query_shape[3]
    attended = Array{A}(undef, query_shape)
    return AttentionWorkspace{T,A}(
        Array{A}(undef, keys, queries, heads, batch), Matrix{A}(undef, keys, queries),
        T === A ? nothing : similar(attended),
        T === A ? nothing : Array{A}(undef, key_shape),
        T === A ? nothing : Array{A}(undef, key_shape),
        attended, T === A ? attended : similar(attended, T),
        similar(attended), Array{A}(undef, key_shape), Array{A}(undef, key_shape))
end

"""
    Attention(width; heads, context_features=nothing, causal=false, precision=Float32)

Features occupy the first axis, tokens the second and examples the last.
Self-attention accepts features or `(features, key_mask)`. Cross-attention
accepts `(query, context)` and an optional third key mask. True keys are visible.
"""
struct Attention{T<:AbstractFloat,A<:AbstractFloat} <: AbstractLayer
    heads::Int
    causal::Bool
    cross_attention::Bool
    scale::A
    query::Dense{T,Identity}
    key::Dense{T,Identity}
    value::Dense{T,Identity}
    projection::Dense{T,Identity}
end

function Attention(width::Int; heads::Int, context_features::Union{Int,Nothing}=nothing,
                   causal::Bool=false, randomization::Bool=true,
                   precision::Type{T}=Float32) where {T<:AbstractFloat}
    checked_precision(T)
    width > 0 && heads > 0 && width % heads == 0 ||
        throw(ArgumentError("attention heads must divide the positive query feature count"))
    cross = context_features !== nothing
    causal && cross && throw(ArgumentError("causal attention requires self-attention"))
    A = accumulation_type(T)
    project(inputs) = Dense(inputs, width; precision=T, randomization)
    context = something(context_features, width)
    return Attention(heads, causal, cross, inv(sqrt(A(width ÷ heads))), project(width),
                     project(context), project(context), project(width))
end

precision_type(::Attention{T}) where {T} = T
layer_children(layer::Attention) =
    (; query=layer.query, key=layer.key, value=layer.value, projection=layer.projection)

sequence_length(input::Nothing) = nothing
function sequence_length(input::Tuple)
    length(input) < 2 && return nothing
    return input[2] isa Colon ? nothing : input[2]
end

function output_shape(layer::Attention, input)
    width = size(layer.query.weights, 2)
    input === nothing && return (width, nothing)
    grouped = layer.cross_attention || all(isnothing, input) || any(x -> x isa Tuple, input)
    parts = grouped ? input : (input,)
    required = layer.cross_attention ? 2 : 1
    required <= length(parts) <= required + 1 || throw(DimensionMismatch("attention input count differs"))
    query = check_shape((width, nothing), first(parts))
    context = layer.cross_attention ? check_shape((size(layer.key.weights, 2), nothing), parts[2]) : query
    if length(parts) > required
        check_shape((sequence_length(context),), last(parts))
    end
    return (width, sequence_length(query))
end

mutable struct AttentionCache{T,A}
    workspace::AttentionWorkspace{T,A}
    AttentionCache{T,A}() where {T,A} = new()
end
layer_cache(::Attention{T,A}) where {T,A} = AttentionCache{T,A}()

function attention_inputs(layer::Attention{T}, input) where {T}
    parts = input isa Tuple ? input : (input,)
    required = layer.cross_attention ? 2 : 1
    required <= length(parts) <= required + 1 || throw(DimensionMismatch("attention input count differs"))
    query = first(parts)
    context = layer.cross_attention ? parts[2] : query
    query isa Array{T,3} && context isa Array{T,3} ||
        throw(ArgumentError("attention requires features × tokens × examples tensors"))
    size(query, 1) == size(layer.query.weights, 2) && size(context, 1) == size(layer.key.weights, 2) ||
        throw(DimensionMismatch("attention feature count differs"))
    size(query, 2) > 0 && size(context, 2) > 0 && size(query, 3) == size(context, 3) ||
        throw(DimensionMismatch("attention needs nonempty sequences and matching batch sizes"))
    mask = length(parts) > required ? last(parts) : nothing
    if length(parts) > required
        mask isa AbstractMatrix{Bool} || throw(ArgumentError("attention key mask must be Boolean"))
        size(mask) == (size(context, 2), size(context, 3)) ||
            throw(DimensionMismatch("attention key mask must have shape (keys, examples)"))
    end
    return query, context, mask
end

from_projection(buffer::Nothing, values) = values
from_projection(buffer::Array, values) = copyto!(buffer, values)

function normalize_scores!(scores::AbstractMatrix{A}, causal, mask, example) where {A}
    @inbounds for query in axes(scores, 2)
        last_key = causal ? query : size(scores, 1)
        peak = A(-Inf)
        for key in axes(scores, 1)
            visible = key <= last_key && (mask === nothing || mask[key, example])
            scores[key, query] = visible ? scores[key, query] : A(-Inf)
            peak = max(peak, scores[key, query])
        end
        if peak == -Inf
            # An example can have no valid keys, for instance when it is all padding.
            for key in axes(scores, 1)
                scores[key, query] = zero(A)
            end
            continue
        end
        total = zero(A)
        for key in axes(scores, 1)
            weight = exp(scores[key, query] - peak)
            scores[key, query] = weight
            total += weight
        end
        @simd for key in axes(scores, 1)
            scores[key, query] /= total
        end
    end
end

function forward!(layer::Attention{T,A}, run::Execution, input, mode) where {T,A}
    query, context, mask = attention_inputs(layer, input)
    Q = execute!(run.children.query, query, mode)
    K = execute!(run.children.key, context, mode)
    V = execute!(run.children.value, context, mode)
    dims = (size(K, 2), size(Q, 2), layer.heads, size(Q, 3))
    if !isdefined(run.cache, :workspace) || size(run.cache.workspace.probabilities) != dims
        run.cache.workspace = AttentionWorkspace(T, A, size(Q), size(K), layer.heads)
    end
    work = run.cache.workspace
    Q = from_projection(work.query, Q)
    K = from_projection(work.key, K)
    V = from_projection(work.value, V)
    head_width = size(Q, 1) ÷ layer.heads
    # Tokens are columns. Each head computes S = KᵀQ / sqrt(d), P = softmax(S), and H = VP.
    @views for example in axes(Q, 3), head in 1:layer.heads
        features = (head - 1) * head_width + 1:head * head_width
        P = work.probabilities[:, :, head, example]
        mul!(P, transpose(K[features, :, example]), Q[features, :, example], layer.scale, zero(A))
        normalize_scores!(P, layer.causal, mask, example)
        mul!(work.attended[features, :, example], V[features, :, example], P)
    end
    T === A || copyto!(work.projection_input, work.attended)
    return execute!(run.children.projection, work.projection_input, mode)
end

function softmax_gradient!(∇P, P)
    @inbounds for query in axes(P, 2)
        projection = zero(eltype(∇P))
        @simd for key in axes(P, 1)
            projection += P[key, query] * ∇P[key, query]
        end
        @simd for key in axes(P, 1)
            ∇P[key, query] = P[key, query] * (∇P[key, query] - projection)
        end
    end
end

function backward!(layer::Attention{T,A}, run::Execution, input, ∇y; compute_parameters) where {T,A}
    work = run.cache.workspace
    ∇H = differentiate!(run.children.projection, ∇y; compute_parameters)
    T === A || (∇H = copyto!(work.attended, ∇H))
    Q = something(work.query, run.children.query.output)
    K = something(work.key, run.children.key.output)
    V = something(work.value, run.children.value.output)
    head_width = size(Q, 1) ÷ layer.heads
    @views for example in axes(Q, 3), head in 1:layer.heads
        features = (head - 1) * head_width + 1:head * head_width
        P = work.probabilities[:, :, head, example]
        ∇H_head = ∇H[features, :, example]
        ∇P = work.score_gradient
        mul!(work.value_gradient[features, :, example], ∇H_head, transpose(P))
        mul!(∇P, transpose(V[features, :, example]), ∇H_head)
        softmax_gradient!(∇P, P)
        ∇S = ∇P
        mul!(work.query_gradient[features, :, example], K[features, :, example],
             ∇S, layer.scale, zero(A))
        mul!(work.key_gradient[features, :, example], Q[features, :, example],
             transpose(∇S), layer.scale, zero(A))
    end
    ∇q = differentiate!(run.children.query, work.query_gradient; compute_parameters)
    ∇k = differentiate!(run.children.key, work.key_gradient; compute_parameters)
    ∇v = differentiate!(run.children.value, work.value_gradient; compute_parameters)
    if layer.cross_attention
        ∇k .+= ∇v
        return length(input) == 3 ? (∇q, ∇k, nothing) : (∇q, ∇k)
    end
    ∇q .= convert.(A, ∇q) .+ convert.(A, ∇k) .+ convert.(A, ∇v)
    return input isa Tuple ? (∇q, nothing) : ∇q
end
