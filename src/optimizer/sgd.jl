module SGD
    using LoopVectorization

    function back_propagation(α::Float64, Current_Layer::Any, Last_Layer::Any, Next_Layer::Any)
        ∇biases = Current_Layer.activation_function.get_∇biases(Current_Layer.value, Next_Layer.propagation_units)
        Current_Layer.get_PU(Current_Layer, ∇biases)

        if Current_Layer.update_weights
            @avx for x in axes(∇biases, 1), y in axes(Last_Layer.output, 1)
                c = 0.0f0
                for b in axes(∇biases, 2)
                    c += ∇biases[x,b]*Last_Layer.output[y,b]
                end
                Current_Layer.weights[x,y] -= c*α*Current_Layer.weights_prop[x,y]
            end
            # Current_Layer.weights -= ∇biases*transpose(Last_Layer.output).*α.*Current_Layer.weights_prop
        end

        if Current_Layer.update_biases
            @avx for i in 1:length(Current_Layer.biases)
                c = 0.0f0
                for b in axes(∇biases, 2)
                    c += ∇biases[i,b]
                end
                Current_Layer.biases[i] -= c*α*Current_Layer.biases_prop[i]
            end
            # Current_Layer.biases -= sum(∇biases, dims=2).*α.*Current_Layer.biases_prop
        end
    end

    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, epochs::Int64=20, batch::Real=32)
        batch_size = ceil(Int64, size(input_data, 2)/batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))

        Threads.@threads for i in 2:length(sequential.layers)
            sequential.layers[i].initializer(sequential.layers[i], 1)
        end

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
                    back_propagation(α, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1])
                end
                if t%ceil(batch_size/50)==0
                    print("=")
                end
                loss += monitor.func(sequential.layers[end-1].output, current_output_data)
            end
            println("] with loss ", loss)
        end
    end
end
