module Minibatch_GD
    function fit(;model::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, epochs::Int64=20, batch::Real=32, mini_batch::Int64=5)
        batch_size = ceil(Int64, size(input_data, 2)/batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))

        model.initialize(model, mini_batch)

        for e in 1:epochs
            print("Epoch ", e, " [")
            Threads.@threads for i in 1:batch_size
                index = rand(1:size(input_data, 2))
                batch_input_data[:,i] = input_data[:,index]
                batch_output_data[:,i] = output_data[:,index]
            end

            current_input_data = zeros(Float32, (size(input_data, 1), mini_batch))
            current_output_data = zeros(Float32, (size(output_data, 1), mini_batch))
            @time begin
                for t in 1:mini_batch:(batch_size÷mini_batch)*mini_batch-(mini_batch-1)
                    current_input_data = batch_input_data[:,t:t+4]
                    current_output_data = batch_output_data[:,t:t+4]

                    if ((t+4)÷mini_batch)%ceil(batch_size/(50*mini_batch))==0
                        print("=")
                    end
                    model.update(model, current_input_data, current_output_data, loss_function, monitor, "Minibatch_GD", α)
                end
                print("] with loss ", model.loss, ", time usage")
                model.loss = 0.0
            end
        end
    end
end
