tensor_signature(x::AbstractArray) = (size(x), eltype(x))
tensor_signature(xs::Union{Tuple,NamedTuple}) = map(tensor_signature, xs)

# Backend extensions provide native replay and any extra warmup it requires.
replay_warmups(backend) = 0
replay_cache(backend) = nothing
warm_replay(f, backend, cache, warmed) = f()
function capture_replay end
backend_replayable(backend, model, example, loss) = true

mutable struct ReplaySession{B,C}
    const backend::B
    const warmups::Int
    cache::C
    buffers::Union{Nothing,Tensor}
    launch::Union{Nothing,Function}
    warmed::Int
    owner::Union{Nothing,Task}
    device::Int
end

function replay_session(backend, eligible)
    warmups = eligible ? replay_warmups(backend) : 0
    warmups == 0 && return nothing
    return ReplaySession(backend, warmups, replay_cache(backend), nothing, nothing, 0, nothing, 0)
end

function invalidate_replay!(session)
    session === nothing && return
    session.warmed < 0 && return
    session.cache = replay_cache(session.backend)
    session.buffers = session.launch = session.owner = nothing
    session.warmed = 0
    return
end

function stage_buffers(backend, x::AbstractArray)
    check_gpu_precision(backend, eltype(x))
    return KA.allocate(backend, eltype(x), size(x))
end
stage_buffers(backend, xs::Union{Tuple,NamedTuple}) = map(x -> stage_buffers(backend, x), xs)

function stage_batch!(buffer::AbstractArray, data::AbstractArray)
    source = device_of(data) === device_of(buffer) ? data : host_array(data)
    return copyto!(buffer, source)
end
stage_batch!(buffers::Tuple, data::Tuple) = foreach(stage_batch!, buffers, data)
stage_batch!(buffers::NamedTuple, data::NamedTuple) = foreach(stage_batch!, buffers, data)

# The callback marks which call is recording and must leave host progress unchanged.
function planned_call!(f, session, data)
    if session === nothing || session.warmed < 0
        f(data, false)
        return false
    end

    device = KA.device(session.backend)
    if session.owner === nothing
        session.buffers = stage_buffers(session.backend, data)
        session.owner, session.device = current_task(), device
    end
    if session.owner !== current_task() || session.device != device
        f(data, false)
        return false
    end

    staged = session.buffers
    stage_batch!(staged, data)
    if session.warmed < session.warmups
        warm_replay(() -> f(staged, false), session.backend, session.cache, session.warmed)
        session.warmed += 1
        return false
    end
    launch = session.launch
    if launch === nothing
        launch = capture_replay(() -> f(staged, true), session.backend, session.cache)
        session.launch = launch
    end
    # A backend can decline to submit an executable if its queue changed.
    if launch === nothing || launch() === false
        session.launch = nothing
        session.warmed = -1
        f(data, false)
        return false
    end
    return true
end

const ReplayLayer = Union{Sequential,Graph,Residual,TransformerBlock,Attention,LayerNorm,BatchNorm2D,
                         GlobalMeanPool2D,MaxPooling2D,Reshape,PositionalEmbedding,Add,Select,
                         Concatenate,Permute,Subtract,PointwiseActivation,Softmax}
const ReplayLoss = Union{MeanSquaredError,QuadraticLoss,AbsoluteLoss,CategoricalCrossEntropy,
                         BinaryCrossEntropy,CrossEntropyWithLogits,BinaryCrossEntropyWithLogits}

# Embedding and indexed target validation read IDs back to the host. Active
# dropout draws a fresh host seed on each forward. Those operations run eagerly.
replayable(layer, training) =
    layer isa ReplayLayer && all(child -> replayable(child, training), layer_children(layer))
replayable(layer::Dropout, training) = !training || iszero(layer.rate)
replayable(layer::Union{Dense,Convolution,UpSampling2D}, training) =
    layer.activation isa Union{PointwiseActivation,Softmax}

# oneAPI currently uses the Dense subset of explicit kernel recording. Uniform precision keeps
# type conversions out of the region, and the bias gradient uses one kernel
# for up to 512 columns.
recordable_dense(layer) = false
recordable_dense(layer::Dense{Float32}) = layer.activation isa PointwiseActivation
recordable_dense(layer::Sequential) = all(recordable_dense, layer.layers)

recordable_workload(model, example, loss) =
    example isa AbstractArray{Float32} && ndims(example) >= 2 && size(example, 1) > 0 &&
    recordable_dense(model) &&
    (loss === nothing || (loss isa Union{MeanSquaredError,QuadraticLoss} &&
                          prod(Base.tail(size(example))) <= 512))

struct InferencePlan{R,P,S,B,C}
    run::R
    reference::P
    signature::S
    backend::B
    replay::C
end

"""
    prepare(model, example; replay=true)

Prepare an inference plan for the example's shape and element type. Pass the
plan to `predict` to reuse its workspace and enable native replay on supported
GPU backends. Copy outputs that must survive the next prediction. Use a separate
plan per concurrent caller.
"""
function prepare(model::AbstractLayer, example; replay::Bool=true)
    reference = parameter_reference(model)
    backend = execution_device(reference, example)
    eligible = replay && replayable(model, false) && backend_replayable(backend, model, example, nothing)
    session = replay_session(backend, eligible)
    return InferencePlan(execution(model), reference, tensor_signature(example), backend, session)
end

function predict(plan::InferencePlan, input)
    backend = execution_device(plan.reference, input)
    session = plan.replay
    same_context = session === nothing || session.owner === nothing ||
                   (backend === plan.backend && session.owner === current_task() &&
                    session.device == KA.device(backend))
    if backend !== plan.backend || tensor_signature(input) != plan.signature || !same_context
        return predict(plan.run.layer, input)
    end
    planned_call!(session, input) do data, _
        execute!(plan.run, batch_on(backend, data))
    end
    return plan.run.output
end

mutable struct TrainingPlan{S,L,I,R}
    const state::S
    const loss::L
    const signature::I
    const replay::R
    forwards::UInt64
    revision::UInt
end

"""
    prepare(state, example, target; loss, replay=true)

Prepare fixed-shape batches for `fit!(plan, data)` or `step!(plan, input, target)`.
The plan reuses the training state's workspace and handles other batch sizes
eagerly. On supported GPU backends, native replay runs the forward and backward
passes for eligible layers and losses. Weight updates and clipping still run each
step. Training that uses active dropout or host-side index validation runs eagerly.
"""
function prepare(state::TrainingState, example, target; loss::AbstractLoss, replay::Bool=true)
    backend = execution_device(parameter_reference(state.model), example)
    eligible = replay && loss isa ReplayLoss && replayable(state.model, true) &&
               backend_replayable(backend, state.model, example, loss)
    session = replay_session(backend, eligible)
    return TrainingPlan(state, loss, tensor_signature((example, target)), session, state.forwards, state.revision)
end

function train_batch!(plan::TrainingPlan, input, target)
    state = plan.state
    state.pending && throw(ArgumentError("complete backward! before another training forward"))
    session = plan.replay
    if session !== nothing &&
       (state.forwards != plan.forwards || state.revision != plan.revision)
        invalidate_replay!(session)
    end
    if tensor_signature((input, target)) == plan.signature &&
       (session === nothing || session.backend === execution_device(parameter_reference(state.model), input))
        replayed = planned_call!(session, (input, target)) do (x, y), recording
            if recording
                # Recording submits no work, so count this batch only once it launches.
                progress = (state.forwards, state.samples, state.microbatches, state.batch, state.pending)
                try
                    train_batch!(state, x, y, plan.loss)
                finally
                    state.forwards, state.samples, state.microbatches, state.batch, state.pending = progress
                end
            else
                train_batch!(state, x, y, plan.loss)
            end
        end
        if replayed
            state.batch = batch_examples(input)
            state.forwards += 1
            state.samples += state.batch
            state.microbatches += 1
        end
    else
        invalidate_replay!(session)
        train_batch!(state, input, target, plan.loss)
    end
    plan.forwards, plan.revision = state.forwards, state.revision
    return state.run.output
end

"""Run one batch with a prepared plan and then update its training state."""
function step!(plan::TrainingPlan, input, target; clip_norm=nothing)
    train_batch!(plan, input, target)
    return step!(plan.state; clip_norm)
end

"""Train with a prepared plan and the loss selected when the plan was created."""
function fit!(plan::TrainingPlan, data; monitor=nothing, epochs::Int=1,
              accumulate::Int=1, clip_norm=nothing, verbose::Bool=true)
    batch!(input, target) = train_batch!(plan, input, target)
    return fit_batches!(batch!, plan.state, data; monitor, epochs, accumulate, clip_norm, verbose)
end
