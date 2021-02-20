module AdaBelief
    function fit(;model::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.001, epochs::Int64=20, batch::Real=32, β₁::Float64=0.9, β₂::Float64=0.999, ϵ::Float64=1e-8)
        batch_size = ceil(Int64, size(input_data, 2)/batch)
        batch_input_data = zeros(Float32, (size(input_data, 1), batch_size))
        batch_output_data = zeros(Float32, (size(output_data, 1), batch_size))

        model.initialize(model, 1)

        for e in 1:epochs
            print("Epoch ", e, " [")
            Threads.@threads for i in 1:batch_size
                index = rand(1:size(input_data, 2))
                batch_input_data[:,i] = input_data[:,index]
                batch_output_data[:,i] = output_data[:,index]
            end

            current_input_data = zeros(Float32, (size(input_data, 1), 1))
            current_output_data = zeros(Float32, (size(output_data, 1), 1))
            @time begin
                for t in 1:batch_size
                    current_input_data[:,1] = batch_input_data[:,t]
                    current_output_data[:,1] = batch_output_data[:,t]

                    if t%ceil(batch_size/50)==0
                        print("=")
                    end
                    model.update(model, current_input_data, current_output_data, loss_function, monitor, "AdaBelief", α, t, β₁, β₂, ϵ)
                end
                print("] with loss ", model.loss, ", time usage")
                model.loss = 0.0
            end
        end
    end
end
