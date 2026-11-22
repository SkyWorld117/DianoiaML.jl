module GA
    using Random: default_rng, shuffle!
    using ..DianoiaML: Residual, initialize!, activate!, batch_buffer, batch_size_for, copy_examples!, sample_count

    function fit(;models::Array, input_data::Array{Float32}, output_data::Array{Float32}, monitor,
                 α::Float64=0.01, num_copy::Int, epochs::Int=20, batch::Real=32, mini_batch::Union{Int,Nothing}=nothing,
                 shuffle::Bool=true, rng=default_rng(), verbose::Bool=true)
        1 <= num_copy <= length(models) || throw(ArgumentError("num_copy must be between 1 and the population size"))
        length(unique(models)) == length(models) || throw(ArgumentError("each genetic candidate must be a distinct model"))
        0 <= α <= 1 || throw(ArgumentError("mutation probability must be in [0, 1]"))
        epochs >= 0 || throw(ArgumentError("epochs must be nonnegative"))
        examples = sample_count(input_data, output_data)
        batch_size = batch_size_for(something(mini_batch, batch), examples)
        steps = cld(examples, batch_size)
        losses = zeros(Float64, length(models))
        fitness = zeros(Float64, num_copy)
        order = collect(1:examples)
        x, y = batch_buffer(input_data, batch_size), batch_buffer(output_data, batch_size)
        tail = rem(examples, batch_size)
        tail_x, tail_y = tail == 0 ? (x, y) : (batch_buffer(input_data, tail), batch_buffer(output_data, tail))
        for model in models
            initialize!(model, batch_size)
        end

        active_size = batch_size
        for epoch in 1:epochs
            loss = 0.0
            shuffle && shuffle!(rng, order)
            for (step, first) in enumerate(1:batch_size:examples)
                count = min(batch_size, examples - first + 1)
                if active_size != count
                    for model in models
                        initialize!(model, count)
                    end
                    active_size = count
                end
                current_x, current_y = count == batch_size ? (x, y) : (tail_x, tail_y)
                copy_examples!(current_x, input_data, order, first)
                copy_examples!(current_y, output_data, order, first)
                for i in eachindex(models)
                    prediction = activate!(models[i], current_x)
                    losses[i] = monitor.func(prediction, current_y)
                end
                rank = sortperm(losses)
                best = losses[rank[1]]
                isfinite(best) || throw(ArgumentError("genetic candidates did not produce a finite loss"))
                loss += best
                for i in 1:num_copy
                    fitness[i] = inv(1 + (losses[rank[i]] - best))
                end
                total = sum(fitness)
                strength = Float32((1 - step / (steps + 1))^5)
                for i in num_copy+1:length(rank)
                    first_parent = models[rank[sample(fitness, total, rng)]]
                    second_parent = models[rank[sample(fitness, total, rng)]]
                    recombine!(models[rank[i]], first_parent, second_parent, α, strength, rng)
                end
            end
            verbose && println("Epoch ", epoch, " mean monitor ", loss / examples)
        end
        return nothing
    end

    function sample(weights, total, rng)
        remaining = rand(rng) * total
        for i in eachindex(weights)
            remaining -= weights[i]
            remaining < 0 && return i
        end
        return findlast(>(0), weights)
    end

    function recombine!(child, parent1, parent2, probability, strength, rng)
        for i in 2:parent1.num_layer+1
            dest, a, b = child.layers[i], parent1.layers[i], parent2.layers[i]
            if dest isa Residual
                recombine!(dest.resnet, a.resnet, b.resnet, probability, strength, rng)
            end
            for field in (:filters, :weights, :biases)
                if hasproperty(dest, field)
                    recombine_array!(getproperty(dest, field), getproperty(a, field), getproperty(b, field), probability, strength, rng)
                end
            end
        end
    end

    function recombine_array!(dest, a, b, probability, strength, rng)
        for i in eachindex(dest, a, b)
            value = a[i] + rand(rng, Float32) * (b[i] - a[i])
            if rand(rng) < probability
                limit = rand(rng, Bool) ? 1f0 : -1f0
                value += (limit - value) * (1f0 - rand(rng, Float32)^strength)
            end
            dest[i] = value
        end
    end
end
