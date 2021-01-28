module Adaptive_Moment_Estimation
    function back_propagation(t::Int64, Vdw::Float64, Sdw::Float64, Vdb::Float64, Sdb::Float64, learning_rate::Float64, batch_size::Int64, Current_Layer::Any, Last_Layer::Any, Next_Layer::Any, β1::Float64, β2::Float64, ϵ::Float64)
        gradient = zeros(size(Current_Layer.value))
        Threads.@threads for b in 1:batch_size
            for j in 1:Current_Layer.layer_size
                if Next_Layer.propagation_units[j,b]!=0
                    gradient[:,b] += Current_Layer.activation_function.diff(Current_Layer.value[:,b], j)*Next_Layer.propagation_units[j,b]
                end
            end
        end
        Threads.@threads for b in 1:batch_size
            for j in 1:Current_Layer.layer_size
                for i in 1:Current_Layer.input_size
                    Current_Layer.propagation_units[j,i,b] = Current_Layer.weights[j,i]*gradient[j,b]
                    Vdw = β1*Vdw+ # need a massive optimize on the old gradient_descent algorithm
                    Current_Layer.weights[j,i] -= (1/batch_size)*learning_rate*gradient[j,b]*Last_Layer.output[i,b]
                end
                Current_Layer.biases[j] -= (1/batch_size)*learning_rate*gradient[j,b]
            end
        end
        Current_Layer.propagation_units = sum(Current_Layer.propagation_units, dims=1)[1,:,:]
        return Vdw, Sdw, Vdb, Sdb
    end

    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, learning_rate::Float64=0.001, epochs::Int64=20, mini_batch::Real=32, β1::Float64=0.9, β2::Float64=0.999, ϵ::Float64=1e-8)
        batch_size = ceil(Int64, size(input_data, 2)/mini_batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))

        Vdw = 0.0
        Sdw = 0.0
        Vdb = 0.0
        Sdb = 0.0

        for e in 1:epochs
            print("Epoch", e)
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
                Vdw, Sdw, Vdb, Sdb = back_propagation(e, Vdw, Sdw, Vdb, Sdb, learning_rate, batch_size, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1], β1, β2, ϵ)
            end
        end
    end
end
