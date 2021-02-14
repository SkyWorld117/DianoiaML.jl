module AdaBelief
    function fit(;sequential::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.001, epochs::Int64=20, batch::Real=32, β₁::Float64=0.9, β₂::Float64=0.999, ϵ::Float64=1e-8)
        batch_size = ceil(Int64, size(input_data, 2)/batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))

        sequential.initializer(sequential, 1)

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
            @time begin
                for t in 1:batch_size
                    current_input_data[:,1] = batch_input_data[:,t]
                    current_output_data[:,1] = batch_output_data[:,t]

                    sequential.activator(sequential, current_input_data)

                    sequential.layers[end].propagation_units = loss_function.prop(sequential.layers[end-1].output, current_output_data)
                    for i in length(sequential.layers)-1:-1:2
                        sequential.layers[i].updater(sequential.layers[i], "Adam", sequential.layers[i-1].output, sequential.layers[i+1].propagation_units, α, t, β₁, β₂, ϵ)
                    end
                    if t%ceil(batch_size/50)==0
                        print("=")
                    end
                    loss += monitor.func(sequential.layers[end-1].output, current_output_data)
                end
                print("] with loss ", loss, ", time usage")
            end
        end
    end
end
