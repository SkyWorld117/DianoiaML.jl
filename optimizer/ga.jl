module GA
    using .Threads, LoopVectorization

    function fit(;models::Array, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, gene_pool::Int64, num_copy::Int64, epochs::Int64=20, batch::Real=32, mini_batch::Int64=5)
        batch_size = ceil(Int64, size(input_data, 2)/batch)*mini_batch
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))

        @threads for i in 1:gene_pool
            models[i].initialize(models[i], mini_batch)
        end

        for e in 1:epochs
            print("Epoch ", e, "\n[")
            @threads for i in 1:batch_size
                index = rand(1:size(input_data, 2))
                batch_input_data[:,i] = input_data[:,index]
                batch_output_data[:,i] = output_data[:,index]
            end

            current_input_data = zeros(Float32, (size(input_data, 1), mini_batch))
            current_output_data = zeros(Float32, (size(output_data, 1), mini_batch))

            loss = 0
            @time begin

                for t in 1:mini_batch:batch_size-mini_batch+1
                    losses = zeros(Float32, gene_pool)
                    current_input_data = batch_input_data[:,t:t+mini_batch-1]
                    current_output_data = batch_output_data[:,t:t+mini_batch-1]

                    if ((t+mini_batch-1)÷mini_batch)%ceil(batch_size/(50*mini_batch))==0
                        print("=")
                    end

                    for i in 1:gene_pool
                        models[i].activate(models[i], current_input_data)
                        losses[i] = monitor.func(models[i].layers[end-1].output, current_output_data)
                    end

                    # Copy
                    new_pool = []
                    for i in 1:num_copy
                        push!(new_pool, splice!(models, argmin(losses)))
                        splice!(losses, argmin(losses))
                    end

                    # Selection -> Recombination -> Mutation
                    mr =
                    for i in 1:gene_pool-num_copy
                        push!(new_pool, recomutation(models[rand(1:length(models))], new_pool[rand(1:num_copy)], mr, α))
                    end

                    models = new_pool
                    loss += minimum(losses)
                end
                print("] with loss ", loss, ", time usage ")
            end
        end
    end

    function recomutation(model₁, model₂, α, mr)
        new_model = model₁
        for i in 1:model₁.num_layer
            if hasproperty(model₁.layers[i], :filters) && hasproperty(model₁.layers[i], :biases)
                @threads for j in eachindex(model₁.layers[i].filters)
                    if rand()<=mr
                        new_model.layers[i].filters[j] = rand(min(model₁.layers[i].filters[j], model₂.layers[i].filters[j]):1.0f-3:max(model₁.layers[i].filters[j], model₂.layers[i].filters[j])) + rand(-α:1.0f-3:α)
                    else
                        new_model.layers[i].filters[j] = rand(min(model₁.layers[i].filters[j], model₂.layers[i].filters[j]):1.0f-3:max(model₁.layers[i].filters[j], model₂.layers[i].filters[j]))
                    end
                end
                @threads for j in eachindex(model₁.layers[i].biases)
                    if rand()<=mr
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j])) + rand(-α:1.0f-3:α)
                    else
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j]))
                    end
                end
            elseif hasproperty(model₁.layers[i], :weights) && hasproperty(model₁.layers[i], :biases)
                @threads for j in eachindex(model₁.layers[i].weights)
                    if rand()<=mr
                        new_model.layers[i].weights[j] = rand(min(model₁.layers[i].weights[j], model₂.layers[i].weights[j]):1.0f-3:max(model₁.layers[i].weights[j], model₂.layers[i].weights[j])) + rand(-α:1.0f-3:α)
                    else
                        new_model.layers[i].weights[j] = rand(min(model₁.layers[i].weights[j], model₂.layers[i].weights[j]):1.0f-3:max(model₁.layers[i].weights[j], model₂.layers[i].weights[j]))
                    end
                end
                @threads for j in eachindex(model₁.layers[i].biases)
                    if rand()<=mr
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j])) + rand(-α:1.0f-3:α)
                    else
                        new_model.layers[i].biases[j] = rand(min(model₁.layers[i].biases[j], model₂.layers[i].biases[j]):1.0f-3:max(model₁.layers[i].biases[j], model₂.layers[i].biases[j]))
                    end
                end
            end
        end
        return new_model
    end
end
