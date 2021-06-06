module GA
    using Polyester, LoopVectorization, VectorizedRNG

    function fit(;models::Array, input_data::Array{Float32}, output_data::Array{Float32}, monitor::Any, α::Float64=0.01, num_copy::Int64, epochs::Int64=20, batch::Real=32, mini_batch::Int64=5)
        gene_pool = length(models)
        input_shape = models[1].layers[2].input_shape
        output_shape = models[1].layers[end].output_shape
        examples = size(input_data)[end]
        batch_size = ceil(Int64, examples/batch)*mini_batch
        current_input_data = zeros(Float32, input_shape..., mini_batch)
        current_output_data = zeros(Float32, output_shape..., mini_batch)

        @batch for i in 1:gene_pool
            models[i].initialize(models[i], mini_batch)
        end

        for e in 1:epochs
            print("Epoch ", e, "\n[")
            loss = 0
            losses = zeros(Float32, gene_pool)
            weights = zeros(Float32, gene_pool)
            @time begin

                for t in 1:mini_batch:batch_size-mini_batch+1
                    @batch for i in 1:mini_batch
                        index = rand(1:examples)
                        selectdim(current_input_data, length(input_shape)+1, i) .= selectdim(input_data, length(input_shape)+1, index)
                        selectdim(current_output_data, length(output_shape)+1, i) .= selectdim(output_data, length(output_shape)+1, index)
                    end

                    if ((t+mini_batch-1)÷mini_batch)%ceil(batch_size/(50*mini_batch))==0
                        print("=")
                    end

                    @time for i in 1:gene_pool
                        models[i].activate(models[i], current_input_data)
                        losses[i] = monitor.func(models[i].layers[end-1].output, current_output_data)
                    end

                    @time for i in 1:gene_pool-num_copy
                        get_weights!(weights, losses)
                        recomutation!(models[argmax(losses)], models[sample(weights)], models[sample(weights)], α, t, batch_size-mini_batch+1)
                        losses[argmax(losses)] = Inf32
                    end

                    loss += minimum(losses)
                end
                print("] with loss ", loss, ", time usage ")
            end
        end
    end

    function get_weights!(weights::Array{Float32}, losses::Array{Float32})
        @avxt for i in eachindex(losses)
            weights[i] = ifelse(losses[i]!=Inf32, 1/losses[i], 0.0f0)
        end
        s = sum(weights)
        @avxt weights ./= s
    end

    function sample(weights)
        r = rand()
        for i in 1:length(weights)
            if weights[i]>=r
                return i
            else
                r -= weights[i]
            end
        end
    end

    function mutation_func(t, v, T)
        if rand(-1:2:1)>0
            return (1.0f0-v)*(1.0f0-rand(0.0f0:1.0f-3:1.0f0)^(1-t/T)^5)
        else
            return -(v+1.0f0)*(1.0f0-rand(0.0f0:1.0f-3:1.0f0)^(1-t/T)^5)
        end
    end

    function recomutation!(new_model, model₁, model₂, α, t, T)
        for i in 1:model₁.num_layer
            if hasproperty(model₁.layers[i], :filters) && hasproperty(model₁.layers[i], :biases)
                @batch for j in eachindex(model₁.layers[i].filters)
                    if rand()<=α
                        new_model.layers[i].filters[j] = rand(min(model₁.layers[i].filters[j], model₂.layers[i].filters[j]):1.0f-3:max(model₁.layers[i].filters[j], model₂.layers[i].filters[j]))
                        new_model.layers[i].filters[j] += mutation_func(t, new_model.layers[i].filters[j], T)
                    else
                        new_model.layers[i].filters[j] = rand(min(model₁.layers[i].filters[j], model₂.layers[i].filters[j]):1.0f-3:max(model₁.layers[i].filters[j], model₂.layers[i].filters[j]))
                    end
                end
                @batch for j in eachindex(model₁.layers[i].biases)
                    if rand()<=α
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j]))
                        new_model.layers[i].biases[j] += mutation_func(t, new_model.layers[i].biases[j], T)
                    else
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j]))
                    end
                end
            elseif hasproperty(model₁.layers[i], :weights) && hasproperty(model₁.layers[i], :biases)
                @batch for j in eachindex(model₁.layers[i].weights)
                    if rand()<=α
                        new_model.layers[i].weights[j] = rand(min(model₁.layers[i].weights[j], model₂.layers[i].weights[j]):1.0f-3:max(model₁.layers[i].weights[j], model₂.layers[i].weights[j]))
                        new_model.layers[i].weights[j] += mutation_func(t, new_model.layers[i].weights[j], T)
                    else
                        new_model.layers[i].weights[j] = rand(min(model₁.layers[i].weights[j], model₂.layers[i].weights[j]):1.0f-3:max(model₁.layers[i].weights[j], model₂.layers[i].weights[j]))
                    end
                end
                @batch for j in eachindex(model₁.layers[i].biases)
                    if rand()<=α
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j]))
                        new_model.layers[i].biases[j] += mutation_func(t, new_model.layers[i].biases[j], T)
                    else
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j]))
                    end
                end
            end
        end
    end
end
