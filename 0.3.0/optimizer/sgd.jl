using .Threads

module SGD
    function back_propagation(α::Float64, batch_size::Int64, Current_Layer::Any, Last_Layer::Any, Next_Layer::Any)
        ∇biases = Current_Layer.activation_function.diff(Current_Layer.value, Next_Layer.propagation_units)
        #println(Current_Layer.activation_function.diff(Current_Layer.value))
        #println(sum(Current_Layer.activation_function.diff(Current_Layer.value, Next_Layer.propagation_units)), "-------", sum(Next_Layer.propagation_units))
        Current_Layer.propagation_units = Current_Layer.get_PU(Current_Layer, ∇biases)

        if Current_Layer.update_weights
            Current_Layer.weights -= (∇biases*transpose(Last_Layer.output).*α).*Current_Layer.weights_prop
        end
        if Current_Layer.update_biases
            Current_Layer.biases -= (sum(∇biases, dims=2).*α).*Current_Layer.biases_prop
        end
    end

    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, epochs::Int64=20, mini_batch::Real=32)
        batch_size = ceil(Int64, size(input_data, 2)/mini_batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))
        for e in 1:epochs
            print("Epoch ", e, " [")
            Threads.@threads for i in 1:batch_size
                index = rand(1:size(input_data, 2))
                batch_input_data[:,i] = input_data[:,index]
                batch_output_data[:,i] = output_data[:,index]
            end
            current_input_data = zeros(Float32, (size(input_data, 1), 1))
            current_output_data = zeros(Float32, (size(output_data, 1), 1))
            loss = 0.0
            for t in 1:batch_size
                current_input_data[:,1] = batch_input_data[:,t]
                current_output_data[:,1] = batch_output_data[:,t]

                @time sequential.activator(sequential, current_input_data)

                sequential.layers[end].propagation_units = loss_function.prop(sequential.layers[end-1].output, current_output_data)
                #println(sequential.layers[end-1].output)
                #println(current_output_data)
                #println(sequential.layers[end].propagation_units)
                #println(sequential.layers[end-1].activation_function.diff(sequential.layers[end-1].value, sequential.layers[end].propagation_units))
                #println(sequential.layers[end-1].value)
                #println(sequential.layers[end-1].activation_function.func(sequential.layers[end-1].value))
                #println(exp.(sequential.layers[end-1].value.-maximum(sequential.layers[end-1].value)), "------", sum(exp.(sequential.layers[end-1].value.-maximum(sequential.layers[end-1].value))))
                #display(sequential.layers[end-1].weights)
                @time for i in length(sequential.layers)-1:-1:2
                    back_propagation(α, batch_size, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1])
                end
                if t%ceil(batch_size/50)==0
                    print("=")
                end
                loss += monitor.func(sequential.layers[end-1].output, current_output_data)
            end
            println("]")
            print(" with loss ", loss)
            sequential.activator(sequential, batch_input_data)
            println("/", monitor.func(sequential.layers[end-1].output, batch_output_data))
        end
    end
end
