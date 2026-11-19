using Random: default_rng, shuffle!

reset_optimizer!(model) = nothing

function reset_optimizer!(layer::Union{Dense,Conv2D,TConv2D})
    for state in (layer.Vdw, layer.Sdw, layer.Vdb, layer.Sdb)
        fill!(state, 0f0)
    end
end

reset_optimizer!(layer::Residual) = reset_optimizer!(layer.resnet)
function reset_optimizer!(model::Union{Sequential,GAN,ResNet})
    for i in 2:length(model.layers)-1
        reset_optimizer!(model.layers[i])
    end
end

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
    size = batch_size_for(batch, examples)
    epochs >= 0 || throw(ArgumentError("epochs must be nonnegative"))
    if !isempty(adaptive_parameters)
        β₁, β₂, ϵ = adaptive_parameters
        0 <= β₁ < 1 && 0 <= β₂ < 1 || throw(ArgumentError("optimizer decay rates must be in [0, 1)"))
        0 < Float32(ϵ) < Inf32 || throw(ArgumentError("optimizer epsilon must be positive and finite"))
    end

    model.initialize(model, size)
    reset_optimizer!(model)
    model.loss = 0.0
    order = collect(1:examples)
    x, y = batch_buffer(input_data, size), batch_buffer(output_data, size)
    tail = rem(examples, size)
    tail_x, tail_y = tail == 0 ? (x, y) : (batch_buffer(input_data, tail), batch_buffer(output_data, tail))
    active_size = size
    step = 0
    for epoch in 1:epochs
        shuffle && shuffle!(rng, order)
        for first in 1:size:examples
            count = min(size, examples - first + 1)
            if active_size != count
                model.initialize(model, count)
                active_size = count
            end
            current_x, current_y = count == size ? (x, y) : (tail_x, tail_y)
            copy_examples!(current_x, input_data, order, first)
            copy_examples!(current_y, output_data, order, first)
            step += 1
            parameters = isempty(adaptive_parameters) ? () : (step, adaptive_parameters...)
            model.update(model, current_x, current_y, loss, monitor, optimizer, α, parameters...)
        end
        verbose && println("Epoch ", epoch, " mean monitor ", model.loss / examples)
        model.loss = 0.0
    end
    return nothing
end
