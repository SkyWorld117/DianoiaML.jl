module Minibatch_GD
    using LoopVectorization, CheapThreads

    function fit(;model::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, epochs::Int64=20, batch::Real=32, mini_batch::Int64=5)
        model.initialize(model, mini_batch)
        input_shape = size(input_data)[1:end-1]
        output_shape = size(output_data)[1:end-1]
        batch_size = ceil(Int64, size(input_data)[end]/batch)*mini_batch

        current_input_data = zeros(Float32, input_shape..., mini_batch)
        current_output_data = zeros(Float32, output_shape..., mini_batch)

        for e in 1:epochs
            print("Epoch ", e, "\n[")
            @time begin
                for t in 1:mini_batch:batch_size-mini_batch+1
                    @batch for i in 1:mini_batch
                        index = rand(1:size(input_data)[end])
                        selectdim(current_input_data, length(input_shape)+1, i) .= selectdim(input_data, length(input_shape)+1, index)
                        selectdim(current_output_data, length(output_shape)+1, i) .= selectdim(output_data, length(output_shape)+1, index)
                    end

                    if ((t+mini_batch-1)÷mini_batch)%ceil(batch_size/(50*mini_batch))==0
                        print("=")
                    end
                    model.update(model, current_input_data, current_output_data, loss_function, monitor, "Minibatch_GD", α)
                end
                print("] with loss ", model.loss, "\nTime usage: ")
                model.loss = 0.0
            end
        end
    end
end
