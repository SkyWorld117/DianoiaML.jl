using Random: Xoshiro, default_rng

"""Configure the parameter update independently of the model and backward pass."""
struct Optimizer{T<:AbstractFloat}
    kind::String
    α::T
    beta1::T
    beta2::T
    epsilon::T

    function Optimizer(kind::Symbol; α::Real=(kind === :SGD ? 1//100 : 1//1000),
                       beta1::Real=9//10, beta2::Real=999//1000, epsilon::Real=1//100000000,
                       precision::Type{T}=promote_type(typeof(float(α)), typeof(float(beta1)),
                                                      typeof(float(beta2)), typeof(float(epsilon)))) where {T<:AbstractFloat}
        kind in (:SGD, :Adam, :AdaBelief) || throw(ArgumentError("unknown optimizer $kind"))
        A = accumulation_type(checked_precision(T))
        rate, decay1, decay2, eps = map(x -> convert(A, x), (α, beta1, beta2, epsilon))
        validate_rate(rate)
        0 <= decay1 < 1 && 0 <= decay2 < 1 || throw(ArgumentError("optimizer decays must be in [0, 1)"))
        0 < eps < Inf || throw(ArgumentError("optimizer epsilon must be positive and finite"))
        new{A}(String(kind), rate, decay1, decay2, eps)
    end
end

function validate_rate(rate)
    0 <= rate < Inf || throw(ArgumentError("learning rate must be nonnegative and finite"))
end

function walk_layers(f, layer, prefix="")
    f(layer, prefix)
    for (i, child) in enumerate(layer_children(layer))
        walk_layers(f, child, prefix * string(i) * ".")
    end
end

function parameter_occurrences(model)
    entries = Tuple{String,Array{<:AbstractFloat},Array{<:AbstractFloat}}[]
    walk_layers(model) do layer, prefix
        for (name, (value, gradient)) in pairs(parameter_gradients(layer))
            size(value) == size(gradient) || throw(DimensionMismatch("parameter and gradient shapes differ at $prefix$name"))
            push!(entries, (prefix * String(name), value, gradient))
        end
    end
    return entries
end

"""Return named references to the model's trainable arrays, including ties."""
parameters(model) = Dict(name => value for (name, value, _) in parameter_occurrences(model))

legacy_moments(layer) = (;)
legacy_moments(layer::Dense) = (; weights=(layer.Vdw, layer.Sdw), biases=(layer.Vdb, layer.Sdb))
legacy_moments(layer::Union{Conv2D,TConv2D}) =
    (; filters=(layer.Vdw, layer.Sdw), biases=(layer.Vdb, layer.Sdb))
legacy_moments(layer::LayerNorm) = (; scale=layer.scale_moments, bias=layer.bias_moments)

# Preserve the direct update! interface and the built-in GAN schedule.
function legacy_step!(layers, indices, optimizer, α, settings, direction)
    updates = IdDict{Any,Tuple}()
    for i in indices
        walk_layers(layers[i]) do layer, _
            for (name, (value, gradient)) in pairs(parameter_gradients(layer))
                if haskey(updates, value)
                    updates[value][1] .+= gradient
                else
                    moments = get(legacy_moments(layer), name, (nothing, nothing))
                    moments[1] === nothing && optimizer ∉ ("SGD", "Minibatch_GD") &&
                        throw(ArgumentError("use TrainingState for adaptive updates on custom parameters"))
                    updates[value] = (copy(gradient), moments...)
                end
            end
        end
    end
    for (value, (gradient, first, second)) in updates
        update_parameters!(optimizer, value, gradient, first, second, α * direction, settings)
    end
end

mutable struct ParameterState{A<:Array{<:AbstractFloat},G<:Array{<:AbstractFloat}}
    value::A
    gradient::G
    first::Union{G,Nothing}
    second::Union{G,Nothing}
    α::AbstractFloat
    trainable::Bool
    step::Int
end

function ParameterState(value::Array{<:AbstractFloat}, optimizer)
    gradient = zeros(accumulation_type(eltype(value)), size(value))
    first = optimizer.kind == "SGD" ? nothing : zero(gradient)
    second = optimizer.kind == "SGD" ? nothing : zero(gradient)
    return ParameterState{typeof(value),typeof(gradient)}(value, gradient, first, second, optimizer.α, true, 0)
end

"""
    TrainingState(model, optimizer; seed)

Own accumulated gradients, optimizer moments, parameter groups, and the
dropout random stream. Create a new state after changing the model topology
or replacing its parameter arrays.
"""
mutable struct TrainingState{M}
    model::M
    optimizer::Optimizer
    slots::Vector{ParameterState}
    names::Vector{String}
    index::IdDict{Any,Int}
    sources::Vector{Tuple{Int,Array{<:AbstractFloat}}}
    seed::UInt64
    forwards::UInt64
    samples::Int
    microbatches::Int
    epochs::Int
    batch::Int
    input::Union{Nothing,Array,Tuple}
end

function TrainingState(model, optimizer::Optimizer; seed::Integer=rand(default_rng(), UInt64))
    validate_layer_tree!(model)
    supports_backward(model) || throw(ArgumentError("every layer must implement the separate backward protocol"))
    slots = ParameterState[]
    names = String[]
    index = IdDict{Any,Int}()
    sources = Tuple{Int,Array{<:AbstractFloat}}[]
    gradient_buffers = Base.IdSet()
    for (name, value, gradient) in parameter_occurrences(model)
        (value === gradient || gradient in gradient_buffers) &&
            throw(ArgumentError("each parameter occurrence needs its own gradient buffer at $name"))
        push!(gradient_buffers, gradient)
        i = get!(index, value) do
            push!(slots, ParameterState(value, optimizer))
            push!(names, name)
            length(slots)
        end
        push!(sources, (i, gradient))
    end
    return TrainingState(model, optimizer, slots, names, index, sources, UInt64(seed), UInt64(0),
                         0, 0, 0, 0, nothing)
end

"""Set the learning rate and/or trainability for a parameter or a model subtree."""
function set_group!(state::TrainingState, group; α=nothing, trainable::Union{Bool,Nothing}=nothing)
    α === nothing || validate_rate(α)
    state.input === nothing && state.samples == 0 ||
        throw(ArgumentError("finish the current update before changing parameter groups"))
    values = group isa Array ? (group,) : (value for (_, value, _) in parameter_occurrences(group))
    indices = unique([state.index[value] for value in values])
    for i in indices
        slot = state.slots[i]
        α === nothing || (slot.α = float(α))
        trainable === nothing || (slot.trainable = trainable)
    end
    return state
end

cpu_batch(x::Array) = x
cpu_batch(x::AbstractArray{<:Real}) = Array(x)
cpu_batch(x::Tuple) = map(cpu_batch, x)

function batch_examples(x::Array)
    ndims(x) > 0 || throw(DimensionMismatch("a batch tensor needs an example axis"))
    return size(x, ndims(x))
end

function batch_examples(xs::Tuple)
    isempty(xs) && throw(ArgumentError("a batch must contain at least one input"))
    batch = batch_examples(first(xs))
    all(x -> batch_examples(x) == batch, xs) || throw(DimensionMismatch("batch input sample counts differ"))
    return batch
end

"""
    forward!(state, input; training=true)

Run a batch of CPU tensors, keeping the forward buffers for one backward call
when training. Floating-point inputs are converted to the model's precision.
"""
function forward!(state::TrainingState, input; training::Bool=true)
    state.input === nothing || throw(ArgumentError("complete backward! before the next forward!"))
    check_parameter_bindings(state)
    data = cpu_batch(float_input(model_precision(state.model), input))
    batch = batch_examples(data)
    batch > 0 || throw(ArgumentError("a batch must contain at least one example"))
    if state.batch != batch
        initialize!(state.model, batch)
        state.batch = batch
    end
    batch_seed = (UInt128(state.seed) << 64) | UInt128(state.forwards)
    mode = training ? TrainingMode(Xoshiro(batch_seed)) : false
    output = activate_layer(state.model, data, mode)
    if training
        state.forwards += 1
        state.input = data
    end
    return output
end

normalize_cotangent(x::LogitGradient) = x
normalize_cotangent(::Nothing) = nothing
normalize_cotangent(x::Tuple) = map(normalize_cotangent, x)
normalize_cotangent(x::AbstractArray) = cpu_batch(x)

function check_parameter_bindings(state)
    occurrences = parameter_occurrences(state.model)
    length(occurrences) == length(state.sources) ||
        throw(ArgumentError("model parameters changed; create a new TrainingState"))
    for ((_, value, derivative), (index, binding)) in zip(occurrences, state.sources)
        (value === state.slots[index].value && derivative === binding) ||
            throw(ArgumentError("model parameters or gradient buffers changed; create a new TrainingState"))
    end
end

check_cotangent(prediction, delta::Nothing) = nothing
check_cotangent(prediction, delta::LogitGradient) = check_cotangent(prediction, delta.data)
function check_cotangent(prediction, delta::Tuple)
    prediction isa Tuple && length(prediction) == length(delta) ||
        throw(DimensionMismatch("output and cotangent tuple lengths differ"))
    foreach(check_cotangent, prediction, delta)
end
function check_cotangent(prediction, delta::AbstractArray)
    prediction isa AbstractArray && size(prediction) == size(delta) ||
        throw(DimensionMismatch("output and cotangent shapes differ"))
end

"""
    backward!(state, upstream)
    backward!(state, target; loss_function)

Propagate explicit cotangents or a built-in loss derivative and accumulate
parameter gradients. The weights and optimizer state are untouched.
"""
function backward!(state::TrainingState, upstream; loss_function=nothing)
    input = state.input
    input === nothing && throw(ArgumentError("call forward! in training mode before backward!"))
    delta = if loss_function === nothing
        normalize_cotangent(upstream)
    elseif loss_function isa Module
        output_gradient!(similar(state.model.output), state.model, normalize_cotangent(upstream), loss_function)
    else
        normalize_cotangent(loss_function(state.model.output, upstream))
    end
    check_cotangent(state.model.output, delta)
    state.input = nothing
    delta === nothing && (delta = zero_cotangent(state.model.output))

    input_gradient = propagate_layer!(state.model, input, delta)
    for (i, gradient) in state.sources
        slot = state.slots[i]
        slot.trainable && (slot.gradient .+= eltype(slot.gradient)(state.batch) .* gradient)
    end
    state.samples += state.batch
    state.microbatches += 1
    return input_gradient
end

"""Return mean accumulated gradients keyed by the first name of each parameter."""
function gradients(state::TrainingState)
    return Dict(name => slot.gradient ./ eltype(slot.gradient)(max(state.samples, 1)) for (name, slot) in zip(state.names, state.slots))
end

function zero_grad!(state::TrainingState)
    for slot in state.slots
        fill!(slot.gradient, zero(eltype(slot.gradient)))
    end
    state.samples = 0
    state.microbatches = 0
    return state
end

"""Apply the mean accumulated gradient once per parameter and clear the gradients."""
function step!(state::TrainingState; clip_norm=nothing)
    state.input === nothing || throw(ArgumentError("complete backward! before step!"))
    state.samples == 0 && return state
    check_parameter_bindings(state)
    scale = 1 // state.samples
    if clip_norm !== nothing
        0 < clip_norm < Inf || throw(ArgumentError("clip_norm must be positive and finite"))
        # Compute the clipping norm in a type that can represent every group.
        N = foldl((t, slot) -> promote_type(t, eltype(slot.gradient)), state.slots; init=Float64)
        norm2 = sum(slot -> slot.trainable ? sum(x -> abs2(N(x)), slot.gradient) : zero(N),
                    state.slots; init=zero(N))
        norm = sqrt(norm2) * scale
        norm > clip_norm && (scale *= clip_norm / norm)
    end
    optimizer = state.optimizer
    for slot in state.slots
        slot.trainable || continue
        slot.gradient .*= eltype(slot.gradient)(scale)
        slot.step += 1
        settings = (slot.step, optimizer.beta1, optimizer.beta2, optimizer.epsilon)
        update_parameters!(optimizer.kind, slot.value, slot.gradient, slot.first, slot.second,
                           slot.α, settings)
    end
    return zero_grad!(state)
end

"""
    fit!(state, batches; loss_function, epochs=1, accumulate=1, clip_norm=nothing)

Train from an iterable yielding `(inputs, targets)`. For multiple epochs use
a restartable iterable or a zero-argument factory. A callable loss receives
`(prediction, target)` and returns the output cotangent(s).
"""
function fit!(state::TrainingState, batches; loss_function, monitor=nothing, epochs::Int=1,
              accumulate::Int=1, clip_norm=nothing, verbose::Bool=true)
    epochs >= 0 || throw(ArgumentError("epochs must be nonnegative"))
    accumulate > 0 || throw(ArgumentError("accumulate must be positive"))
    for _ in 1:epochs
        examples = 0
        score = 0.0
        iterator = applicable(batches) ? batches() : batches
        for (input, target) in iterator
            prediction = forward!(state, input)
            backward!(state, target; loss_function)
            examples += state.batch
            if monitor !== nothing
                score += monitor isa Module ? monitor.func(prediction, cotangent_as(prediction, normalize_cotangent(target))) : monitor(prediction, target)
            end
            if state.microbatches >= accumulate
                step!(state; clip_norm)
            end
        end
        examples > 0 || throw(ArgumentError("the batch iterator produced no examples"))
        step!(state; clip_norm)
        state.epochs += 1
        if verbose && monitor !== nothing
            println("Epoch ", state.epochs, " mean monitor ", score / examples)
        end
    end
    return state
end
