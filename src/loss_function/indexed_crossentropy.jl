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
        if size(z, 1) >= INDEXED_LOSS_GROUP
            launch!(indexed_partition_gpu!, values, z, ids, scale, Val(false);
                    ndrange=INDEXED_LOSS_GROUP * length(ids), groupsize=INDEXED_LOSS_GROUP)
        else
            launch!(indexed_loss_gpu!, values, z, ids, scale)
        end
        return sum_map(identity, values, zero(scale))
    end
    return sum(i -> indexed_loss(z, ids, i, scale), eachindex(ids))
end

function loss_gradient!(dest, ::IndexedCrossEntropyWithLogits, output, target::AbstractArray{<:Integer})
    size(dest) == size(output) || throw(DimensionMismatch("logits and gradient shapes differ"))
    z, ids, scale = indexed_logits(output, target)
    ∇z = reshape(dest, size(z))
    if gpu_array(output)
        if size(z, 1) >= INDEXED_LOSS_GROUP
            launch!(indexed_partition_gpu!, ∇z, z, ids, scale, Val(true);
                    ndrange=INDEXED_LOSS_GROUP * length(ids), groupsize=INDEXED_LOSS_GROUP)
        else
            launch!(indexed_gradient_gpu!, ∇z, z, ids, scale; ndrange=length(ids))
        end
    else
        for i in eachindex(ids)
            indexed_gradient!(∇z, z, ids, i, scale)
        end
    end
    return dest
end

const INDEXED_LOSS_GROUP = 256

@kernel inbounds=true function indexed_partition_gpu!(dest, z, ids, scale::A, ::Val{gradient}) where {A,gradient}
    lane = @index(Local, Linear)
    col = @index(Group, Linear)
    scratch = @localmem A (INDEXED_LOSS_GROUP,)
    peak = A(-Inf)
    for i in lane:INDEXED_LOSS_GROUP:size(z, 1)
        peak = max(peak, A(z[i, col]))
    end
    scratch[lane] = peak
    @synchronize
    for offset in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= offset
            scratch[lane] = max(scratch[lane], scratch[lane + offset])
        end
        @synchronize
    end
    peak = scratch[1]
    @synchronize
    total = zero(A)
    for i in lane:INDEXED_LOSS_GROUP:size(z, 1)
        total += exp(A(z[i, col]) - peak)
    end
    scratch[lane] = total
    @synchronize
    for offset in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= offset
            scratch[lane] += scratch[lane + offset]
        end
        @synchronize
    end
    id = ids[col]
    if gradient
        factor = scale / scratch[1]
        for i in lane:INDEXED_LOSS_GROUP:size(z, 1)
            dest[i, col] = factor * exp(A(z[i, col]) - peak) - (i == id) * scale
        end
    elseif lane == 1
        dest[col] = scale * (log(scratch[1]) - (A(z[id, col]) - peak))
    end
end

@kernel inbounds=true function indexed_loss_gpu!(values, z, ids, scale)
    i = @index(Global, Linear)
    values[i] = indexed_loss(z, ids, i, scale)
end

@kernel inbounds=true function indexed_gradient_gpu!(dest, z, ids, scale)
    i = @index(Global, Linear)
    indexed_gradient!(dest, z, ids, i, scale)
end
