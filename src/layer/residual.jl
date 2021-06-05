module ResidualM
    using LoopVectorization
    include("../network/resnet.jl")
    using .resnet:ResNet
    
    mutable struct Residual
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple

        resnet::Any

        output::Array{Float32}
        δ::Array{Float32}

        function Residual(;input_shape::Tuple, resnet::Any)
            new(save_Residual, load_Residual, activate_Residual, init_Residual, update_Residual, input_shape, input_shape, resnet)
        end
    end

    function init_Residual(layer::Residual, mini_batch::Int64)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
        layer.resnet.initialize(layer.resnet, mini_batch)
    end

    function activate_Residual(layer::Residual, input::Array{Float32})
        layer.resnet.activate(layer.resnet, input)
        t = layer.resnet.num_layer+1
        @avxt for i in eachindex(input)
            layer.output[i] = layer.resnet.layers[t].output[i] + input[i]
        end
    end

    function update_Residual(layer::Residual, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        layer.resnet.layers[end].δ = δₗ₊₁
        layer.resnet.update(layer.resnet, optimizer, α, parameters...)
        @avxt for i in eachindex(layer.δ)
            layer.δ[i] = δₗ₊₁[i]*(1 + layer.resnet.layers[2].δ[i])
        end
    end

    function save_Residual(layer::Residual, file::Any, id::String)
        write(file, id, "Residual")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"num_layer", layer.resnet.num_layer)
        for i in 2:layer.resnet.num_layer+1
            layer.resnet.layers[i].save_layer(layer.resnet.layers[i], file, id*"resnet"*string(i-1))
        end
    end

    function load_Residual(layer::Residual, file::Any, id::String)
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        resnet = ResNet()
        num_layer = read(file, id*"num_layer")
        for i in 1:num_layer
            s = read(file, id*"resnet"*string(i))
            layer = Symbol(s)
            mod = Symbol(s*"M")
            args = eval(mod).get_args(file, string(i))
            try
                activation_function = Symbol(read(file, id*"resnet"*string(i)*"activation_function"))
                resnet.add_layer(resnet, eval(layer); args..., activation_function=eval(activation_function))
            catch KeyError
                resnet.add_layer(resnet, eval(layer); args...)
            end
            resnet.layers[end].load_layer(resnet.layers[end], file, id*"resnet"*string(i))
        end
        return (input_shape=input_shape, resnet=resnet)
    end
end