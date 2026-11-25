function device_attention_workspace(Q, K, ::Type{A}, dims) where {A}
    probabilities = similar(Q, A, dims)
    return DeviceAttentionWorkspace(probabilities, similar(probabilities), similar(Q),
                                    similar(Q, A), similar(K, A), similar(K, A))
end

@kernel inbounds=true function attention_scores_gpu!(scores, Q, K, width, scale::A) where {A}
    key, query, head, b = @index(Global, NTuple)
    total = zero(A)
    for i in 1:width
        f = (head-1) * width + i
        total = muladd(A(K[f, key, b]), A(Q[f, query, b]), total)
    end
    scores[key, query, head, b] = total * scale
end

@kernel inbounds=true function attention_softmax_gpu!(P, mask, causal)
    query, head, b = @index(Global, NTuple)
    A = eltype(P)
    peak = A(-Inf)
    for key in axes(P, 1)
        visible = (!causal || key <= query) && (mask === nothing || mask[key, b])
        score = visible ? P[key, query, head, b] : A(-Inf)
        P[key, query, head, b] = score
        peak = max(peak, score)
    end
    total = zero(A)
    for key in axes(P, 1)
        weight = peak == -Inf ? zero(A) : exp(P[key, query, head, b] - peak)
        P[key, query, head, b] = weight
        total += weight
    end
    if !iszero(total)
        for key in axes(P, 1)
            P[key, query, head, b] /= total
        end
    end
end

@kernel inbounds=true function attention_values_gpu!(H, V, P, width)
    f, query, b = @index(Global, NTuple)
    A = eltype(P)
    head = cld(f, width)
    total = zero(A)
    for key in axes(P, 1)
        total = muladd(A(V[f, key, b]), P[key, query, head, b], total)
    end
    H[f, query, b] = total
end

const ATTENTION_GROUP = (16, 4, 1, 1)
const SEQUENCE_GROUP = (16, 4, 1)

function gpu_attention_forward!(layer::Attention{T,A}, run, query, context, mask, mode) where {T,A}
    Q = execute!(run.children.query, query, mode)
    K = execute!(run.children.key, context, mode)
    V = execute!(run.children.value, context, mode)
    dims = (size(K, 2), size(Q, 2), layer.heads, size(Q, 3))
    if !isdefined(run.cache, :workspace) || size(run.cache.workspace.probabilities) != dims
        run.cache.workspace = device_attention_workspace(Q, K, A, dims)
    end
    work = run.cache.workspace
    P = work.probabilities
    width = size(Q, 1) ÷ layer.heads
    launch!(attention_scores_gpu!, P, Q, K, width, layer.scale; ndrange=dims, groupsize=ATTENTION_GROUP)
    launch!(attention_softmax_gpu!, P, mask, layer.causal; ndrange=Base.tail(dims), groupsize=SEQUENCE_GROUP)
    launch!(attention_values_gpu!, work.attended, V, P, width; ndrange=size(Q), groupsize=SEQUENCE_GROUP)
    return execute!(run.children.projection, work.attended, mode)
end

@kernel inbounds=true function attention_probability_gradient_gpu!(∇P, ∇H, V, width)
    key, query, head, b = @index(Global, NTuple)
    A = eltype(∇P)
    total = zero(A)
    for i in 1:width
        f = (head-1) * width + i
        total = muladd(A(V[f, key, b]), A(∇H[f, query, b]), total)
    end
    ∇P[key, query, head, b] = total
end

@kernel inbounds=true function attention_softmax_backward_gpu!(∇P, P)
    query, head, b = @index(Global, NTuple)
    projection = zero(eltype(P))
    for key in axes(P, 1)
        projection += P[key, query, head, b] * ∇P[key, query, head, b]
    end
    for key in axes(P, 1)
        ∇P[key, query, head, b] = P[key, query, head, b] * (∇P[key, query, head, b] - projection)
    end
end

@kernel inbounds=true function attention_query_gradient_gpu!(∇Q, K, ∇S, width, scale::A) where {A}
    f, query, b = @index(Global, NTuple)
    head = cld(f, width)
    total = zero(A)
    for key in axes(∇S, 1)
        total = muladd(A(K[f, key, b]), ∇S[key, query, head, b], total)
    end
    ∇Q[f, query, b] = total * scale
end

@kernel inbounds=true function attention_context_gradient_gpu!(∇K, ∇V, Q, ∇H, ∇S, P, width, scale::A) where {A}
    f, key, b = @index(Global, NTuple)
    head = cld(f, width)
    key_total, value_total = zero(A), zero(A)
    for query in axes(P, 2)
        key_total = muladd(A(Q[f, query, b]), ∇S[key, query, head, b], key_total)
        value_total = muladd(A(∇H[f, query, b]), P[key, query, head, b], value_total)
    end
    ∇K[f, key, b] = key_total * scale
    ∇V[f, key, b] = value_total
end

function gpu_attention_backward!(layer::Attention{T,A}, run, input, ∇y, compute_parameters) where {T,A}
    work = run.cache.workspace
    ∇H = differentiate!(run.children.projection, ∇y; compute_parameters)
    Q, K, V = run.children.query.output, run.children.key.output, run.children.value.output
    P, ∇S = work.probabilities, work.score_gradient
    width = size(Q, 1) ÷ layer.heads
    launch!(attention_probability_gradient_gpu!, ∇S, ∇H, V, width;
            ndrange=size(P), groupsize=ATTENTION_GROUP)
    launch!(attention_softmax_backward_gpu!, ∇S, P; ndrange=Base.tail(size(P)), groupsize=SEQUENCE_GROUP)
    launch!(attention_query_gradient_gpu!, work.query_gradient, K, ∇S, width, layer.scale;
            ndrange=size(Q), groupsize=SEQUENCE_GROUP)
    launch!(attention_context_gradient_gpu!, work.key_gradient, work.value_gradient, Q, ∇H, ∇S, P,
            width, layer.scale; ndrange=size(K), groupsize=SEQUENCE_GROUP)

    ∇q = differentiate!(run.children.query, work.query_gradient; compute_parameters)
    ∇k = differentiate!(run.children.key, work.key_gradient; compute_parameters)
    ∇v = differentiate!(run.children.value, work.value_gradient; compute_parameters)
    if layer.cross_attention
        broadcast_tensor!(+, ∇k, ∇k, ∇v)
        return length(input) == 3 ? (∇q, ∇k, nothing) : (∇q, ∇k)
    end
    broadcast_tensor!((q, k, v) -> A(q) + A(k) + A(v), ∇q, ∇q, ∇k, ∇v)
    return input isa Tuple ? (∇q, nothing) : ∇q
end
