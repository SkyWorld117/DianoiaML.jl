mutable struct ParameterState{A<:AbstractArray{<:AbstractFloat},G<:AbstractArray{<:AbstractFloat}}
    value::A
    gradient::G
    first::Union{Nothing,G}
    second::Union{Nothing,G}
    lr::AbstractFloat
    trainable::Bool
    step::Int
end

function ParameterState(value, optimizer)
    gradient = zeros_like(value, accumulation_type(eltype(value)), size(value))
    count = moment_count(optimizer)
    first = count >= 1 ? zero(gradient) : nothing
    second = count >= 2 ? zero(gradient) : nothing
    return ParameterState(value, gradient, first, second, optimizer.lr, true, 0)
end

"""
    TrainingState(model, optimizer; seed)

Own every invocation's forward and backward buffers, accumulated gradients,
optimizer moments and dropout random stream. The model owns its parameters.
"""
mutable struct TrainingState{M,E}
    const model::M
    run::E
    optimizer::AbstractOptimizer
    slots::Vector{ParameterState}
    names::Vector{String}
    index::IdDict{Any,Int}
    sources::Vector{Tuple{Int,AbstractArray{<:AbstractFloat}}}
    aliases::Vector{Tuple{String,Int}}
    seed::UInt64
    forwards::UInt64
    samples::Int
    microbatches::Int
    epochs::Int
    batch::Int
    pending::Bool
end

function TrainingState(model::AbstractLayer, optimizer::AbstractOptimizer;
                       seed::Integer=rand(default_rng(), UInt64))
    run = execution(model)
    slots = ParameterState[]
    names = String[]
    index = IdDict{Any,Int}()
    sources = Tuple{Int,AbstractArray{<:AbstractFloat}}[]
    aliases = Tuple{String,Int}[]
    buffers = Base.IdSet()
    walk_executions(run) do node, prefix
        values = layer_parameters(node.layer)
        gradients = layer_gradients(node.layer, node.cache)
        keys(values) == keys(gradients) || throw(ArgumentError("parameter and gradient names differ at $prefix"))
        for (name, value) in pairs(values)
            local_grad = gradients[name]
            size(value) == size(local_grad) || throw(DimensionMismatch("parameter and gradient shapes differ at $prefix$name"))
            (value === local_grad || local_grad in buffers) &&
                throw(ArgumentError("each invocation needs its own gradient buffer at $prefix$name"))
            push!(buffers, local_grad)
            path = prefix * string(name)
            i = get!(index, value) do
                push!(slots, ParameterState(value, optimizer))
                push!(names, path)
                length(slots)
            end
            push!(sources, (i, local_grad))
            push!(aliases, (path, i))
        end
    end
    return TrainingState(model, run, optimizer, slots, names, index, sources, aliases, UInt64(seed),
                         UInt64(0), 0, 0, 0, 0, false)
end

function set_group!(state::TrainingState, group; lr=nothing, trainable::Union{Bool,Nothing}=nothing)
    lr === nothing || validate_rate(lr)
    !state.pending && state.samples == 0 || throw(ArgumentError("finish the current update before changing groups"))
    values_to_update = group isa AbstractArray ? (group,) : values(parameters(group))
    indices = unique([state.index[value] for value in values_to_update])
    for i in indices
        slot = state.slots[i]
        lr === nothing || (slot.lr = float(lr))
        trainable === nothing || (slot.trainable = trainable)
    end
    return state
end

"""Run a training forward pass and retain data for one backward call."""
function forward!(state::TrainingState, input)
    state.pending && throw(ArgumentError("complete backward! before another training forward"))
    reference = isempty(state.slots) ? nothing : first(state.slots).value
    data = batch_on(execution_device(reference, input), input)
    count = batch_examples(data)
    count > 0 || throw(ArgumentError("a batch needs at least one example"))
    seed = (UInt128(state.seed) << 64) | UInt128(state.forwards)
    result = execute!(state.run, data, TrainingMode(Xoshiro(seed)))
    state.batch = count
    state.forwards += 1
    state.pending = true
    return result
end

check_cotangent(prediction, ::Nothing) = nothing
function check_cotangent(prediction, delta::AbstractArray)
    prediction isa AbstractArray && size(prediction) == size(delta) ||
        throw(DimensionMismatch("output and cotangent shapes differ"))
end
function check_cotangent(prediction, delta::Union{Tuple,NamedTuple})
    typeof(keys(prediction)) === typeof(keys(delta)) && keys(prediction) == keys(delta) ||
        throw(DimensionMismatch("output and cotangent structures differ"))
    foreach(check_cotangent, prediction, delta)
end

complete_cotangent(prediction, ::Nothing) = zero_cotangent(prediction)
complete_cotangent(prediction, delta) = batch_on(device_of(prediction), delta)
complete_cotangent(prediction::Tuple, delta::Tuple) = map(complete_cotangent, prediction, delta)
complete_cotangent(prediction::NamedTuple, delta::NamedTuple) = map(complete_cotangent, prediction, delta)

"""
    backward!(state, upstream)
    backward!(state, target; loss)

Propagate explicit cotangents or differentiate a loss. Parameter gradients are
accumulated by example count and weights are updated separately by step!.
"""
function backward!(state::TrainingState, upstream; loss::Union{Nothing,AbstractLoss}=nothing)
    return backpropagate!(state, upstream, loss; accumulate=true)
end

function backpropagate!(state, upstream, loss; accumulate)
    state.pending || throw(ArgumentError("call forward! before backward!"))
    output = state.run.output
    delta = loss === nothing ? upstream :
            loss_gradient!(similar(output), loss, output, batch_on(device_of(output), upstream))
    check_cotangent(output, delta)
    state.pending = false
    derivative = differentiate!(state.run, complete_cotangent(output, delta); compute_parameters=accumulate)
    if accumulate
        for (i, gradient) in state.sources
            slot = state.slots[i]
            slot.trainable && (slot.gradient .+= eltype(slot.gradient)(state.batch) .* gradient)
        end
        state.samples += state.batch
        state.microbatches += 1
    end
    return derivative
end

"""Return accumulated mean gradients under the first name of each shared parameter."""
function gradients(state::TrainingState)
    return Dict(name => slot.gradient ./ eltype(slot.gradient)(max(state.samples, 1)) for (name, slot) in zip(state.names, state.slots))
end

function zero_grad!(state::TrainingState)
    for slot in state.slots
        fill!(slot.gradient, zero(eltype(slot.gradient)))
    end
    state.samples = state.microbatches = 0
    return state
end

function step!(state::TrainingState; clip_norm=nothing)
    state.pending && throw(ArgumentError("complete backward! before step!"))
    state.samples == 0 && return state
    scale = 1 // state.samples
    if clip_norm !== nothing
        0 < clip_norm < Inf || throw(ArgumentError("clip_norm must be positive and finite"))
        backend = isempty(state.slots) ? nothing : device_of(first(state.slots).gradient)
        N = foldl((t, slot) -> promote_type(t, eltype(slot.gradient)), state.slots;
                  init=wide_float(backend))
        norm2 = sum(slot -> slot.trainable ? sum_map(SquaredMagnitude{N}(), slot.gradient, zero(N)) : zero(N),
                    state.slots; init=zero(N))
        norm = sqrt(norm2) * scale
        norm > clip_norm && (scale *= clip_norm / norm)
    end
    for slot in state.slots
        slot.trainable || continue
        slot.gradient .*= eltype(slot.gradient)(scale)
        slot.step += 1
        update_parameters!(state.optimizer, slot.value, slot.gradient, slot.first, slot.second,
                           slot.lr, slot.step)
    end
    return zero_grad!(state)
end

"""Train from an iterable of (inputs, targets), or a factory that creates one."""
function fit!(state::TrainingState, data; loss::AbstractLoss, monitor=nothing, epochs::Int=1,
              accumulate::Int=1, clip_norm=nothing, verbose::Bool=true)
    epochs >= 0 || throw(ArgumentError("epochs must be nonnegative"))
    accumulate > 0 || throw(ArgumentError("accumulate must be positive"))
    for _ in 1:epochs
        count, score = 0, 0.0
        for (input, target) in (applicable(data) ? data() : data)
            prediction = forward!(state, input)
            backward!(state, target; loss)
            count += state.batch
            monitor === nothing || (score += monitor(prediction, target))
            state.microbatches < accumulate || step!(state; clip_norm)
        end
        count > 0 || throw(ArgumentError("the batch iterator produced no examples"))
        step!(state; clip_norm)
        state.epochs += 1
        verbose && monitor !== nothing && println("Epoch ", state.epochs, " mean monitor ", score / count)
    end
    return state
end
