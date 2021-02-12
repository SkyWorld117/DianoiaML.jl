module AdaBelief
    function back_propagation(t::Int64, α::Float64, Current_Layer::Any, Last_Layer::Any, Next_Layer::Any, β₁::Float64, β₂::Float64, ϵ::Float64)
        ∇biases = Current_Layer.activation_function.get_∇biases(Current_Layer.value, Next_Layer.propagation_units)
        Current_Layer.propagation_units = Current_Layer.get_PU(Current_Layer, ∇biases)

        if Current_Layer.update_weights
            ∇weights = ∇biases*transpose(Last_Layer.output)
            Current_Layer.Vdw = Current_Layer.Vdw .* β₁ + ∇weights .* (1-β₁)
            weights_belief = ∇weights-Current_Layer.Vdw
            Current_Layer.Sdw = Current_Layer.Sdw .* β₂ + (weights_belief .* weights_belief) .* (1-β₂) .+ ϵ
            Current_Layer.weights -= α.*(Current_Layer.Vdw./(1-β₁^t))./(sqrt.(Current_Layer.Sdw./(1-β₂^t)).+ϵ).*Current_Layer.weights_prop
        end

        if Current_Layer.update_biases
            ∇biases = sum(∇biases, dims=2)
            Current_Layer.Vdb = Current_Layer.Vdb .* β₁ + ∇biases .* (1-β₁)
            biases_belief = ∇biases-Current_Layer.Vdb
            Current_Layer.Sdb = Current_Layer.Sdb .* β₂ + (biases_belief .* biases_belief) .* (1-β₂)
            Current_Layer.biases -= α.*(Current_Layer.Vdb./(1-β₁^t))./(sqrt.(Current_Layer.Sdb./(1-β₂^t)).+ϵ).*Current_Layer.biases_prop
        end
    end

    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.001, epochs::Int64=20, batch::Real=32, β₁::Float64=0.9, β₂::Float64=0.999, ϵ::Float64=1e-8)
        batch_size = ceil(Int64, size(input_data, 2)/batch)
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

                sequential.activator(sequential, current_input_data)

                sequential.layers[end].propagation_units = loss_function.prop(sequential.layers[end-1].output, current_output_data)
                for i in length(sequential.layers)-1:-1:2
                    back_propagation(t, α, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1], β₁, β₂, ϵ)
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
