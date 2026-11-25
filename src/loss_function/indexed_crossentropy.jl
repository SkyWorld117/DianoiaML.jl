"""Cross-entropy for class-first logits and one-based integer labels.

A label is supplied for each position per example. The loss averages the positions
within an example and sums across the batch, like the other training losses.
"""
struct IndexedCrossEntropyWithLogits <: AbstractLoss end

function indexed_logits(output, target::AbstractArray{<:Integer})
    size(output)[2:end] == size(target) && size(output, 1) > 0 && !isempty(target) ||
        throw(DimensionMismatch("expected one nonempty integer label tensor for the trailing logit axes"))
    ids = vec(compatible_target(output, target))
    lo, hi = extrema(ids)
    1 <= lo <= hi <= size(output, 1) || throw(ArgumentError("class labels must be in 1:size(logits, 1)"))
    A = loss_type(output)
    scale = inv(A(length(target) ÷ size(target, ndims(target))))
    return reshape(output, size(output, 1), :), ids, scale
end

@inline function indexed_partition(z, col, ::Type{A}) where {A}
    peak = logit_peak(z, col, A)
    total = zero(A)
    @inbounds for i in axes(z, 1)
        total += exp(A(z[i, col]) - peak)
    end
    return peak, total
end

@inline function indexed_loss(z, ids, col, scale::A) where {A}
    peak, total = indexed_partition(z, col, A)
    return scale * (log(total) - (@inbounds A(z[ids[col], col]) - peak))
end

@inline function indexed_gradient!(dest, z, ids, col, scale::A) where {A}
    peak, total = indexed_partition(z, col, A)
    id = @inbounds ids[col]
    @inbounds for i in axes(z, 1)
        dest[i, col] = scale * (exp(A(z[i, col]) - peak) / total - (i == id))
    end
    return
end

function loss_value(::IndexedCrossEntropyWithLogits, output, target::AbstractArray{<:Integer})
    z, ids, scale = indexed_logits(output, target)
    if gpu_array(output)
        values = similar(output, typeof(scale), length(ids))
        launch!(indexed_loss_gpu!, values, z, ids, scale)
        return sum_map(identity, values, zero(scale))
    end
    return sum(i -> indexed_loss(z, ids, i, scale), eachindex(ids))
end

function loss_gradient!(dest, ::IndexedCrossEntropyWithLogits, output, target::AbstractArray{<:Integer})
    size(dest) == size(output) || throw(DimensionMismatch("logits and gradient shapes differ"))
    z, ids, scale = indexed_logits(output, target)
    ∇z = reshape(dest, size(z))
    if gpu_array(output)
        launch!(indexed_gradient_gpu!, ∇z, z, ids, scale; ndrange=length(ids))
    else
        for i in eachindex(ids)
            indexed_gradient!(∇z, z, ids, i, scale)
        end
    end
    return dest
end

@kernel inbounds=true function indexed_loss_gpu!(values, z, ids, scale)
    i = @index(Global, Linear)
    values[i] = indexed_loss(z, ids, i, scale)
end

@kernel inbounds=true function indexed_gradient_gpu!(dest, z, ids, scale)
    i = @index(Global, Linear)
    indexed_gradient!(dest, z, ids, i, scale)
end
