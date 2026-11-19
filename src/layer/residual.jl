module ResidualM
    using ..resnet: ResNet
    import ..DianoiaML: activate_layer

    mutable struct Residual
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple

        resnet::ResNet

        output::Array{Float32}
        δ::Array{Float32}

        function Residual(;input_shape::Tuple, resnet::ResNet)
            new(save_Residual, load_Residual, activate_Residual, init_Residual, update_Residual, input_shape, input_shape, resnet)
        end
    end

    function init_Residual(layer::Residual, mini_batch::Int)
        layer.resnet.initialize(layer.resnet, mini_batch)
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    activate_layer(layer::Residual, input, training) = activate_Residual(layer, input; training)

    function activate_Residual(layer::Residual, input::Array{Float32}; training::Bool=false)
        branch = layer.resnet.activate(layer.resnet, input; training)
        size(branch) == size(input) || throw(DimensionMismatch("the residual branch must preserve the input shape"))
        layer.output .= branch .+ input
    end

    function update_Residual(layer::Residual, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int=1)
        copyto!(layer.resnet.layers[end].δ, δₗ₊₁)
        layer.resnet.update(layer.resnet, optimizer, α, parameters...; direction)
        layer.δ .= δₗ₊₁ .+ layer.resnet.layers[2].δ
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
        parent = parentmodule(@__MODULE__)
        for i in 1:num_layer
            parent.load_layer!(resnet, resnet.add_layer, file, id * "resnet" * string(i))
        end
        return (input_shape=input_shape, resnet=resnet)
    end
end
