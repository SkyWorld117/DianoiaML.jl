using .Threads

module Minibatch_Gradient_Descent
    function back_propagation(α::Float64, batch_size::Int64, Current_Layer::Any, Last_Layer::Any, Next_Layer::Any)
        ∇biases = Current_Layer.activation_function.diff(Current_Layer.value).*Next_Layer.propagation_units
        Current_Layer.propagation_units = Current_Layer.get_PU(Current_Layer, ∇biases)

        if Current_Layer.update_weights
            Current_Layer.weights -= (∇biases*transpose(Last_Layer.output).*(α/batch_size)).*Current_Layer.weights_prop
        end
        if Current_Layer.update_biases
            Current_Layer.biases -= (sum(∇biases, dims=2).*(α/batch_size)).*Current_Layer.biases_prop
        end
    end

    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, epochs::Int64=20, mini_batch::Real=32)
        batch_size = ceil(Int64, size(input_data, 2)/mini_batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))
        for e in 1:epochs
            print("Epoch ", e)
            Threads.@threads for i in 1:batch_size
                index = rand(1:size(input_data, 2))
                batch_input_data[:,i] = input_data[:,index]
                batch_output_data[:,i] = output_data[:,index]
            end
            sequential.activator(sequential, batch_input_data)
            sequential.layers[end].propagation_units = loss_function.prop(sequential.layers[end-1].output, batch_output_data)
            println(" with loss ", sum(monitor.func(sequential.layers[end-1].output, batch_output_data)))

            for i in length(sequential.layers)-1:-1:2
                println(" Back propagating...Layer", i-1)
                back_propagation(α, batch_size, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1])
            end
        end
    end
end
