using .Threads

module Adam
    function back_propagation(t::Int64, α::Float64, batch_size::Int64, Current_Layer::Any, Last_Layer::Any, Next_Layer::Any, β1::Float64, β2::Float64, ϵ::Float64)
        ∇biases = Current_Layer.activation_function.diff(Current_Layer.value).*Next_Layer.propagation_units
        Current_Layer.propagation_units = Current_Layer.get_PU(Current_Layer, ∇biases)

        if Current_Layer.update_weights
            ∇weights = ∇biases*transpose(Last_Layer.output)./batch_size
            Current_Layer.Vdw = Current_Layer.Vdw .* β1 + ∇weights .* (1-β1)
            Current_Layer.Sdw = Current_Layer.Sdw .* β2 + broadcast(*, ∇weights, ∇weights) .* (1-β2)
            Current_Layer.weights -= α.*(Current_Layer.Vdw./(1-β1^t))./(sqrt.(Current_Layer.Sdw./(1-β2^t)).+ϵ).*Current_Layer.weights_prop
        end

        if Current_Layer.update_biases
            ∇biases = sum(∇biases, dims=2)./batch_size
            Current_Layer.Vdb = Current_Layer.Vdb .* β1 + ∇biases .* (1-β1)
            Current_Layer.Sdb = Current_Layer.Sdb .* β2 + broadcast(*, ∇biases, ∇biases) .* (1-β2)
            Current_Layer.biases -= α.*(Current_Layer.Vdb./(1-β1^t))./(sqrt.(Current_Layer.Sdb./(1-β2^t)).+ϵ).*Current_Layer.biases_prop
        end
    end

    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.001, epochs::Int64=20, mini_batch::Real=32, β1::Float64=0.9, β2::Float64=0.999, ϵ::Float64=1e-8)
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

            println(" with loss ", monitor.func(sequential.layers[end-1].output, batch_output_data))
            sequential.layers[end].propagation_units = loss_function.prop(sequential.layers[end-1].output, batch_output_data)

            for i in length(sequential.layers)-1:-1:2
                println(" Back propagating...Layer", i-1)
                back_propagation(e, α, batch_size, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1], β1, β2, ϵ)
            end
        end
    end
end
