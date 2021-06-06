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

                    for i in 1:gene_pool
                        models[i].activate(models[i], current_input_data)
                        losses[i] = monitor.func(models[i].layers[end-1].output, current_output_data)
                    end

                    for i in 1:gene_pool-num_copy
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
            if hasproperty(new_model.layers[i], :filters)
                @avxt temp = model₂.layers[i].filters .- model₁.layers[i].filters
                rand!(local_rng(), new_model.layers[i].filters, VectorizedRNG.StaticInt(0), model₁.layers[i].filters, temp)
                rand!(local_rng(), temp)
                @avxt for j in eachindex(temp)
                    new_model.layers[i].filters[j] = ifelse(temp[j]<=α, ifelse(temp[j]<=0.5, (1.0f0-new_model.layers[i].filters[j])*(1.0f0-temp[j]^(1-t/T)^5), -(new_model.layers[i].filters[j]+1.0f0)*(1.0f0-temp[j]^(1-t/T)^5)), new_model.layers[i].filters[j])
                end
            end

            if hasproperty(new_model.layers[i], :weights)
                @avxt temp = model₂.layers[i].weights .- model₁.layers[i].weights
                rand!(local_rng(), new_model.layers[i].weights, VectorizedRNG.StaticInt(0), model₁.layers[i].weights, temp)
                rand!(local_rng(), temp)
                @avxt for j in eachindex(temp)
                    new_model.layers[i].weights[j] = ifelse(temp[j]<=α, ifelse(temp[j]<=0.5, (1.0f0-new_model.layers[i].weights[j])*(1.0f0-temp[j]^(1-t/T)^5), -(new_model.layers[i].weights[j]+1.0f0)*(1.0f0-temp[j]^(1-t/T)^5)), new_model.layers[i].weights[j])
                end
            end

            if hasproperty(new_model.layers[i], :biases)
                @avxt temp = model₂.layers[i].biases .- model₁.layers[i].biases
                rand!(local_rng(), new_model.layers[i].biases, VectorizedRNG.StaticInt(0), model₁.layers[i].biases, temp)
                rand!(local_rng(), temp)
                @avxt for j in eachindex(temp)
                    new_model.layers[i].biases[j] = ifelse(temp[j]<=α, ifelse(temp[j]<=0.5, (1.0f0-new_model.layers[i].biases[j])*(1.0f0-temp[j]^(1-t/T)^5), -(new_model.layers[i].biases[j]+1.0f0)*(1.0f0-temp[j]^(1-t/T)^5)), new_model.layers[i].biases[j])
                end
            end
        end
    end
end
