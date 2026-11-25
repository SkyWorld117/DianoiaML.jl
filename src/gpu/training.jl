compatible_target(output, target) = gpu_array(output) ? batch_on(device_of(output), target) : target

@kernel inbounds=true function squared_gradient_gpu!(gradient, prediction, target, scale::A) where {A}
    i = @index(Global, Linear)
    gradient[i] = scale * (A(prediction[i]) - A(target[i]))
end

@kernel inbounds=true function accumulate_gradient_gpu!(gradient, source, count)
    i = @index(Global, Linear)
    gradient[i] += count * source[i]
end

@inline point_loss(::Union{QuadraticLoss,MeanSquaredError}, x, y, ε) = abs2(x - y)
@inline point_loss(::AbsoluteLoss, x, y, ε) = abs(x - y)
@inline point_loss(::CategoricalCrossEntropy, p, y, ε) = -y * log(max(p, ε))
@inline point_loss(::BinaryCrossEntropy, p, y, ε) =
    -y * log(max(p, ε)) - (1 - y) * log(max(1 - p, ε))
@inline point_loss(::BinaryCrossEntropyWithLogits, z, y, ε) =
    max(z, zero(z)) - z * y + log1p(exp(-abs(z)))

@kernel inbounds=true function point_loss_gpu!(output, loss, x, y, ε)
    i = @index(Global, Linear)
    A = eltype(output)
    output[i] = point_loss(loss, A(x[i]), A(y[i]), ε)
end

function gpu_loss_value(loss, output, target)
    A = loss_type(output)
    ε = A(probability_floor(eltype(output)))
    values = similar(output, A)
    launch!(point_loss_gpu!, values, loss, output, compatible_target(output, target), ε)
    total = sum_map(identity, values, zero(A))
    return loss isa MeanSquaredError ? total * squared_scale(loss, output) : total
end

@inline function logit_peak(z, col, ::Type{A}) where {A}
    peak = A(-Inf)
    @inbounds for i in axes(z, 1)
        peak = max(peak, A(z[i, col]))
    end
    return peak
end

@kernel inbounds=true function crossentropy_value_gpu!(values, z, y)
    col = @index(Global, Linear)
    A = eltype(values)
    peak = logit_peak(z, col, A)
    Z = zero(A)
    for i in axes(z, 1)
        Z += exp(A(z[i, col]) - peak)
    end
    total = zero(A)
    log_Z = log(Z)
    for i in axes(z, 1)
        total += A(y[i, col]) * (log_Z - (A(z[i, col]) - peak))
    end
    values[col] = total
end

function gpu_loss_value(::CrossEntropyWithLogits, output, target)
    A = loss_type(output)
    z, y = map(x -> reshape(x, size(output, 1), :), (output, compatible_target(output, target)))
    values = similar(output, A, (size(z, 2),))
    launch!(crossentropy_value_gpu!, values, z, y)
    return sum_map(identity, values, zero(A))
end

@kernel inbounds=true function crossentropy_gradient_gpu!(∇z, z, y, ::Type{A}) where {A}
    col = @index(Global, Linear)
    peak = logit_peak(z, col, A)
    Z, mass = zero(A), zero(A)
    for i in axes(z, 1)
        Z += exp(A(z[i, col]) - peak)
        mass += A(y[i, col])
    end
    for i in axes(z, 1)
        ∇z[i, col] = (mass / Z) * exp(A(z[i, col]) - peak) - A(y[i, col])
    end
end

function gpu_crossentropy_gradient!(dest, output, target)
    ∇z, z, y = map(x -> reshape(x, size(output, 1), :), (dest, output, compatible_target(output, target)))
    launch!(crossentropy_gradient_gpu!, ∇z, z, y, loss_type(output); ndrange=size(z, 2))
    return dest
end

@kernel inbounds=true function classification_gpu!(errors, predicted, expected, threshold)
    col = @index(Global, Linear)
    if size(predicted, 1) == 1
        errors[col] = (predicted[1, col] >= threshold) != (expected[1, col] >= 0.5f0)
    else
        actual, target = 1, 1
        for i in 2:size(predicted, 1)
            actual = isless(predicted[actual, col], predicted[i, col]) ? i : actual
            target = isless(expected[target, col], expected[i, col]) ? i : target
        end
        errors[col] = actual != target
    end
end

function gpu_classification(monitor, output, target)
    predicted, expected = map(x -> reshape(x, size(output, 1), :), (output, compatible_target(output, target)))
    errors = similar(output, Int32, (size(predicted, 2),))
    threshold = wide_float(device_of(output))(monitor.threshold)
    launch!(classification_gpu!, errors, predicted, expected, threshold)
    return sum_map(identity, errors, 0)
end

@kernel inbounds=true function momentum_gpu!(w, ∇w, velocity, α, μ)
    i = @index(Global, Linear)
    v = μ * velocity[i] + ∇w[i]
    velocity[i] = v
    w[i] = w[i] - α * v
end

@kernel inbounds=true function rmsprop_gpu!(w, ∇w, squares, α, ε, ρ, gain)
    i = @index(Global, Linear)
    g = ∇w[i]
    v = ρ * squares[i] + gain * abs2(g)
    squares[i] = v
    w[i] = w[i] - α * g / (sqrt(v) + ε)
end

@kernel inbounds=true function adaptive_optimizer_gpu!(w, ∇w, first, second, α̂, ε, scale,
                                                       decay₁, decay₂, gain₁, gain₂, correction, belief)
    i = @index(Global, Linear)
    g = ∇w[i]
    m = decay₁ * first[i] + gain₁ * g
    deviation = ifelse(belief, g - m, g)
    v = decay₂ * second[i] + gain₂ * deviation^2
    first[i], second[i] = m, v
    w[i] = scale * w[i] - α̂ * m / (sqrt(v * correction) + ε)
end

const ADAMW_GROUP_SIZE = 8
const ADAMW_SMALL_PARAMETER = 2048

function adamw_groups(optimizer, slots)
    optimizer isa AdamW && !isempty(slots) || return nothing
    backend = device_of(first(slots).value)
    backend !== nothing && all(slot -> device_of(slot.value) === backend, slots) || return nothing

    # Separately registered views of the same storage must keep their
    # sequential update order. Tied arrays already share a single slot.
    allocations = Set{UInt}()
    for slot in slots
        ids = Base.dataids(slot.value)
        any(in(allocations), ids) && return nothing
        union!(allocations, ids)
    end

    groups = AdamWGroup[]
    small = Dict{Type,Vector{ParameterState}}()
    for slot in slots
        if length(slot.value) <= ADAMW_SMALL_PARAMETER
            key = Tuple{typeof(vec(slot.value)),typeof(vec(slot.gradient))}
            push!(get!(() -> ParameterState[], small, key), slot)
        else
            push!(groups, adamw_group((slot,)))
        end
    end
    for same_type in values(small), start in 1:ADAMW_GROUP_SIZE:length(same_type)
        stop = min(start + ADAMW_GROUP_SIZE - 1, length(same_type))
        push!(groups, adamw_group(Tuple(same_type[start:stop])))
    end
    return groups
end

function adamw_group(slots)
    buffers = map(slot -> (vec(slot.value), vec(slot.gradient), vec(slot.first), vec(slot.second)), slots)
    return AdamWGroup(slots, buffers, maximum(buffer -> length(first(buffer)), buffers))
end

function adamw_group_step!(group::AdamWGroup, optimizer, mean_scale)
    settings = map(group.slots) do slot
        A = eltype(slot.gradient)
        slot.step += slot.trainable
        coefficients = adaptive_coefficients(optimizer, A, slot.lr, max(1, slot.step))
        return (slot.trainable, A(mean_scale), coefficients)
    end
    if group.width > 0
        backend = device_of(first(first(group.buffers)))
        tensors = length(group.buffers)
        lanes = tensors == 1 ? 256 : 128
        adamw_group_gpu!(backend, (lanes, 1))(group.buffers, settings; ndrange=(group.width, tensors))
    end
    return
end

@kernel inbounds=true function adamw_group_gpu!(buffers, settings)
    i, tensor = @index(Global, NTuple)
    w, ∇w, first, second = buffers[tensor]
    if i <= length(w)
        trainable, mean_scale, coefficients = settings[tensor]
        if trainable
            α̂, ε, scale, decay₁, decay₂, gain₁, gain₂, correction = coefficients
            g = mean_scale * ∇w[i]
            m = decay₁ * first[i] + gain₁ * g
            v = decay₂ * second[i] + gain₂ * g^2
            first[i], second[i] = m, v
            w[i] = scale * w[i] - α̂ * m / (sqrt(v * correction) + ε)
        end
        ∇w[i] = zero(eltype(∇w))
    end
end

@kernel inbounds=true function recombine_gpu!(dest, a, b, probability::A, strength, seed) where {A}
    i = @index(Global, Linear)
    start = 3 * i
    value = A(a[i]) + uniform_sample(A, random_bits(seed, start)) * (A(b[i]) - A(a[i]))
    choice = random_bits(seed, start + 1)
    if uniform_sample(A, choice) < probability
        limit = isodd(choice) ? one(A) : -one(A)
        value += (limit - value) * (one(A) - uniform_sample(A, random_bits(seed, start + 2))^strength)
    end
    dest[i] = value
end
