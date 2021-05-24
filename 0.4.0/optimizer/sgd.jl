module SGD
    using LoopVectorization

    function fit(;model::Any, input_data::Array{Float32}, output_data::Array{Float32}, loss_function::Any, monitor::Any, α::Float64=0.01, epochs::Int64=20, batch::Real=32)
        model.initialize(model, 1)
        input_shape = size(input_data)[1:end-1]
        output_shape = size(output_data)[1:end-1]
        batch_size = ceil(Int64, size(input_data, length(input_shape)+1)/batch)

        current_input_data = zeros(Float32, input_shape..., 1)
        current_output_data = zeros(Float32, output_shape..., 1)

        for e in 1:epochs
            print("Epoch ", e, "\n[")
            @time begin
                for t in 1:batch_size
                    index = rand(1:size(input_data)[end])
                    current_input_data .= reshape(selectdim(input_data, length(input_shape)+1, index), input_shape..., 1)
                    current_output_data .= reshape(selectdim(output_data, length(output_shape)+1, index), output_shape..., 1)

                    if t%ceil(batch_size/50)==0
                        print("=")
                    end
                    model.update(model, current_input_data, current_output_data, loss_function, monitor, "SGD", α)
                end
                print("] with loss ", model.loss, "\nTime usage: ")
                model.loss = 0.0
            end
        end
    end
end
