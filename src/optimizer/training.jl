batch_examples(x::AbstractArray) = size(x, ndims(x))

function batch_examples(xs::Union{Tuple,NamedTuple})
    isempty(xs) && throw(ArgumentError("a batch needs at least one array"))
    count = batch_examples(first(xs))
    all(x -> batch_examples(x) == count, xs) || throw(DimensionMismatch("batch sample counts differ"))
    return count
end

struct Batches{X,Y,R}
    inputs::X
    targets::Y
    batch_size::Int
    count::Int
    shuffle::Bool
    rng::R
end

"""
    batches(inputs, [targets]; batch_size, shuffle=true, rng)

Iterate over views with examples on the last axis. A new iteration reshuffles
when enabled, and the final batch includes any remaining examples.
"""
function batches(inputs, targets=nothing; batch_size::Int, shuffle::Bool=true, rng=default_rng())
    count = batch_examples(inputs)
    count > 0 && batch_size > 0 || throw(ArgumentError("batch size and sample count must be positive"))
    targets === nothing || batch_examples(targets) == count ||
        throw(DimensionMismatch("inputs and targets have different numbers of examples"))
    return Batches(inputs, targets, batch_size, count, shuffle, rng)
end

Base.length(data::Batches) = cld(data.count, data.batch_size)
Base.IteratorEltype(::Type{<:Batches}) = Base.EltypeUnknown()

select_examples(x::Array, indices) = selectdim(x, ndims(x), indices)
select_examples(x::AbstractArray, indices) =
    gpu_array(x) ? gather_examples(x, indices) : selectdim(x, ndims(x), indices)
select_examples(inputs::Union{Tuple,NamedTuple}, indices) = map(x -> select_examples(x, indices), inputs)

function Base.iterate(data::Batches)
    order = data.shuffle ? shuffle!(data.rng, collect(1:data.count)) : (1:data.count)
    return iterate(data, (order, 1))
end

function Base.iterate(data::Batches, (order, start))
    start > data.count && return nothing
    finish = start + min(data.batch_size, data.count - start + 1) - 1
    indices = view(order, start:finish)
    inputs = select_examples(data.inputs, indices)
    result = data.targets === nothing ? inputs : (inputs, select_examples(data.targets, indices))
    return result, (order, finish + 1)
end
