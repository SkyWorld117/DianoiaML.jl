using Random: shuffle!

reset_optimizer!(model) = nothing

function reset_optimizer!(layer::Union{Dense,Conv2D,TConv2D})
    for state in (layer.Vdw, layer.Sdw, layer.Vdb, layer.Sdb)
        fill!(state, zero(eltype(state)))
    end
end

function reset_optimizer!(layer::LayerNorm)
    for state in (layer.scale_moments..., layer.bias_moments...)
        fill!(state, zero(eltype(state)))
    end
end

reset_optimizer!(model::Union{AbstractLayerStack,GAN,Residual,Attention}) =
    foreach(reset_optimizer!, layer_children(model))

function sample_count(input, target)
    examples = size(input, ndims(input))
    examples > 0 || throw(ArgumentError("training data must contain at least one example"))
    examples == size(target, ndims(target)) || throw(DimensionMismatch("input and target sample counts differ"))
    return examples
end

function batch_size_for(batch, examples)
    isinteger(batch) && batch > 0 || throw(ArgumentError("batch size must be a positive integer"))
    return Int(min(batch, examples))
end

function batch_buffer(data, batch)
    return similar(data, size(data)[1:end-1]..., batch)
end

function copy_examples!(buffer, data, order, first)
    samples = size(buffer, ndims(buffer))
    features = length(buffer) ÷ samples
    for b in 1:samples
        index = order[first + b - 1]
        copyto!(buffer, (b - 1) * features + 1, data, (index - 1) * features + 1, features)
    end
end

function fit_gradient!(model, input_data, output_data, loss, monitor, optimizer, α, epochs, batch,
                       adaptive_parameters; shuffle, rng, verbose)
    examples = sample_count(input_data, output_data)
    batch_size = batch_size_for(batch, examples)
    epochs >= 0 || throw(ArgumentError("epochs must be nonnegative"))
    if !isempty(adaptive_parameters)
        β₁, β₂, ϵ = adaptive_parameters
        0 <= β₁ < 1 && 0 <= β₂ < 1 || throw(ArgumentError("optimizer decay rates must be in [0, 1)"))
        0 < ϵ < Inf || throw(ArgumentError("optimizer epsilon must be positive and finite"))
    end

    initialize!(model, batch_size)
    reset_optimizer!(model)
    hasproperty(model, :loss) && (model.loss = 0.0)
    state = if supports_backward(model)
        settings = isempty(adaptive_parameters) ? (; α) :
            (; α, beta1=adaptive_parameters[1], beta2=adaptive_parameters[2], epsilon=adaptive_parameters[3])
        state = TrainingState(model, Optimizer(Symbol(optimizer); settings...))
        state.batch = batch_size
        state
    else
        nothing
    end
    order = collect(1:examples)
    x, y = batch_buffer(input_data, batch_size), batch_buffer(output_data, batch_size)
    tail = rem(examples, batch_size)
    tail_x, tail_y = tail == 0 ? (x, y) : (batch_buffer(input_data, tail), batch_buffer(output_data, tail))
    active_size = batch_size
    step = 0
    for epoch in 1:epochs
        score = 0.0
        shuffle && shuffle!(rng, order)
        for first in 1:batch_size:examples
            count = min(batch_size, examples - first + 1)
            if state === nothing && active_size != count
                initialize!(model, count)
                active_size = count
            end
            current_x, current_y = count == batch_size ? (x, y) : (tail_x, tail_y)
            copy_examples!(current_x, input_data, order, first)
            copy_examples!(current_y, output_data, order, first)
            if state === nothing
                step += 1
                parameters = isempty(adaptive_parameters) ? () : (step, adaptive_parameters...)
                update!(model, current_x, current_y, loss, monitor, optimizer, α, parameters...)
            else
                prediction = forward!(state, current_x)
                backward!(state, current_y; loss_function=loss)
                step!(state)
                score += monitor.func(prediction, cotangent_as(prediction, current_y))
            end
        end
        if state === nothing
            score = model.loss
            model.loss = 0.0
        end
        verbose && println("Epoch ", epoch, " mean monitor ", score / examples)
    end
    return nothing
end
