module convolutional_2d
    mutable struct Convolutional_2D
        activator::Any

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        kernel::Array{Float32}
        weights::Array{Float32}
        biases::Array{Float32}

        input2D_size::Tuple{Int64, Int64}
        step_x::Int64
        step_y::Int64

        value::Array{Float32}
        output::Array{Float32}
        propagation_units::Array{Float32} # need massive update for layers

        Vdw::Array{Float32}
        Sdw::Array{Float32}
        Vdb::Array{Float32}
        Sdb::Array{Float32}

        function Convolutional_2D(;input_size::Tuple{Int64, Int64}, kernel_size::Tuple{Int64, Int64}, step_x::Int64=1, step_y::Int64=1, activation_function::Module, randomization::Bool=true, reload::Bool=false)
            if reload
                return new(activate_Conv2D)
            end

            biases = zeros(Float32, input_size[1]*input_size[2])
            weights = zeros(Float32, (Int64(((input_size[1]-kernel_size[1])/step_x+1)*((input_size[2]-kernel[2])/step_y+1)),input_size[1]*input_size[2]))
            if not randomization
                kernel = zeros(Float32, kernel_size)
            else
                kernel = 0.1*rand(Float32, kernel_size)
                for i in 0:size(weights, 1)-1
                    index = Int64(step_x*(i%((input_size[1]-kernel_size[1])/step_x+1)) + input_size[1]*step_y*(i÷((input_size[1]-kernel_size[1])/x+1)))
                    for j in 1:size(kernel, 1)
                        for k in 1:size(kernel, 2)
                            index += 1
                            weights[i+1,index] = kernel[j,k]
                        end
                        index += input_size[1]-kernel[1]
                    end
                end
            end
            new(activate_Conv2D, size(weights, 2), size(weights, 1), activation_function, kernel, weights, biases, input_size, step_x, step_y)
        end
    end

    function activate_Conv2D(layer::Convolutional_2D, input::Array{Float32})
        layer.value = layer.weights*input .+ layer.biases
        layer.output = layer.activation_function.func(layer.value)
        layer.propagation_units = zeros(Float32, (layer.layer_size, layer.input_size, size(input, 2)))

        layer.Vdw = zeros(Float32, size(layer.weights))
        layer.Sdw = zeros(Float32, size(layer.weights))
        layer.Vdb = zeros(Float32, size(layer.biases))
        layer.Sdb = zeros(Float32, size(layer.biases))
    end
end

# Source: https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
