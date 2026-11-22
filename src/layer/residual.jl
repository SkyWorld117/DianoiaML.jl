module ResidualM
    using ..resnet: ResNet
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, load_layer!, activate_layer

    mutable struct Residual{N} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        resnet::ResNet

        output::Array{Float32,N}
        δ::Array{Float32,N}

        function Residual(;input_shape::Tuple, resnet::ResNet)
            new{length(input_shape) + 1}(input_shape, input_shape, resnet)
        end
    end

    function initialize!(layer::Residual, mini_batch::Int)
        initialize!(layer.resnet, mini_batch)
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    activate_layer(layer::Residual, input, training) = activate!(layer, input; training)

    function activate!(layer::Residual, input::Array{Float32}; training::Bool=false)
        branch = activate!(layer.resnet, input; training)
        size(branch) == size(input) || throw(DimensionMismatch("the residual branch must preserve the input shape"))
        layer.output .= branch .+ input
    end

    function update!(layer::Residual, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int=1)
        copyto!(layer.resnet.layers[end].δ, δₗ₊₁)
        update!(layer.resnet, optimizer, α, parameters...; direction)
        layer.δ .= δₗ₊₁ .+ layer.resnet.layers[2].δ
    end

    function save_layer(layer::Residual, file, id::String)
        write(file, id, "Residual")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"num_layer", layer.resnet.num_layer)
        for i in 2:layer.resnet.num_layer+1
            save_layer(layer.resnet.layers[i], file, id * "resnet" * string(i - 1))
        end
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        resnet = ResNet()
        num_layer = read(file, id*"num_layer")
        for i in 1:num_layer
            load_layer!(resnet, resnet.add_layer, file, id * "resnet" * string(i))
        end
        return (input_shape=input_shape, resnet=resnet)
    end
end
