"""Select and recombine a population of models using a score to minimize."""
function fit_population!(models, data; monitor, keep::Int, mutation::Real=0.01,
                         epochs::Int=1, rng=default_rng(), verbose::Bool=true)
    1 <= keep <= length(models) || throw(ArgumentError("keep must be between one and the population size"))
    0 <= mutation <= 1 && epochs >= 0 || throw(ArgumentError("invalid mutation rate or epoch count"))
    arrays = map(parameters, models)
    names = sort!(collect(keys(first(arrays))))
    layout = population_bindings(first(arrays), names)
    independent = unique(layout)
    seen = Base.IdSet()
    for values in arrays
        keys(values) == keys(first(arrays)) || throw(ArgumentError("candidate parameters differ"))
        population_bindings(values, names) == layout || throw(ArgumentError("candidate parameter ties differ"))
        for name in independent
            shape = size(first(arrays)[name])
            value = values[name]
            size(value) == shape || throw(DimensionMismatch("candidate parameter shapes differ"))
            value in seen && throw(ArgumentError("candidates must own distinct parameters"))
            push!(seen, value)
        end
    end
    runs = map(execution, models)
    A = foldl(promote_type, (eltype(value) for value in seen); init=Float64)
    scores = zeros(A, length(models))
    fitness = zeros(A, keep)
    for epoch in 1:epochs
        count, score = 0, zero(A)
        for (iteration, (input, target)) in enumerate(data)
            for i in eachindex(runs)
                backend = execution_device(parameter_reference(models[i]), input)
                scores[i] = monitor(execute!(runs[i], batch_on(backend, input)), target)
            end
            rank = sortperm(scores)
            best = scores[rank[1]]
            isfinite(best) || throw(ArgumentError("no candidate produced a finite score"))
            fitness .= inv.(1 .+ scores[rank[1:keep]] .- best)
            total = sum(fitness)
            strength = (1 - A(iteration) / (length(data) + 1))^5
            for i in keep+1:length(rank)
                a = arrays[rank[select_candidate(fitness, total, rng)]]
                b = arrays[rank[select_candidate(fitness, total, rng)]]
                child = arrays[rank[i]]
                for name in independent
                    recombine_array!(child[name], a[name], b[name], mutation, strength, rng)
                end
            end
            count += batch_examples(input)
            score += best
        end
        count > 0 || throw(ArgumentError("the batch iterator produced no examples"))
        verbose && println("Epoch ", epoch, " mean monitor ", score / count)
    end
    return models
end

function population_bindings(arrays, names)
    first_names = IdDict{Any,String}()
    for name in names
        get!(first_names, arrays[name], name)
    end
    return map(name -> first_names[arrays[name]], names)
end

function select_candidate(weights, total, rng)
    remaining = rand(rng) * total
    for i in eachindex(weights)
        remaining -= weights[i]
        remaining < 0 && return i
    end
    return findlast(>(0), weights)
end

function recombine_array!(dest, a, b, probability, strength, rng)
    A = accumulation_type(eltype(dest))
    R = random_precision(A)
    gpu_array(dest) && return launch!(recombine_gpu!, dest, a, b, A(probability), A(strength), rand(rng, UInt64))
    for i in eachindex(dest, a, b)
        value = A(a[i]) + A(rand(rng, R)) * (A(b[i]) - A(a[i]))
        if rand(rng) < probability
            limit = rand(rng, Bool) ? one(A) : -one(A)
            value += (limit - value) * (one(A) - A(rand(rng, R))^A(strength))
        end
        dest[i] = value
    end
end
