module ResidualM
    using ..resnet: ResNet
    using ..DianoiaML: AbstractLayer, AbstractLayerStack
    import ..DianoiaML
    import ..DianoiaML: initialize!, activate!, update!, save_layer, activate_layer, layer_children, validate_connections

    mutable struct Residual{N,S<:AbstractLayerStack} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        resnet::S

        output::Array{Float32,N}
        δ::Array{Float32,N}

        function Residual(;input_shape::Tuple, resnet::AbstractLayerStack)
            new{length(input_shape) + 1, typeof(resnet)}(input_shape, input_shape, resnet)
        end
    end

    """
        Residual(network)

    Add a network's output to its input. The nested network must preserve the
    input shape.
    """
    function Residual(network::AbstractLayerStack)
        network.num_layer > 0 || throw(ArgumentError("a residual network requires at least one layer"))
        layer = Residual(; input_shape=network.input_shape, resnet=network)
        validate_connections(layer)
        return layer
    end

    layer_children(layer::Residual) = (layer.resnet,)

    function validate_connections(layer::Residual)
        branch = layer.resnet
        if branch.num_layer > 0 && !(branch.input_shape == layer.input_shape == branch.output_shape)
            throw(DimensionMismatch("the residual branch must preserve the input shape"))
        end
    end

    function initialize!(layer::Residual, mini_batch::Int)
        initialize!(layer.resnet, mini_batch)
        validate_connections(layer)
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
        branch_delta = update!(layer.resnet, optimizer, input, δₗ₊₁, α, parameters, direction)
        layer.δ .= δₗ₊₁ .+ branch_delta
    end

    function save_layer(layer::Residual, file, id::String)
        write(file, id, "Residual")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"num_layer", layer.resnet.num_layer)
        if !(layer.resnet isa ResNet)
            write(file, id * "network_type", string(nameof(typeof(layer.resnet))))
        end
        DianoiaML.save_stack_layers(file, layer.resnet, id * "resnet")
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        type_key = id * "network_type"
        resnet = haskey(file, type_key) ? DianoiaML.stack_from_name(read(file, type_key)) : ResNet()
        num_layer = read(file, id*"num_layer")
        DianoiaML.load_stack_layers!(resnet, file, id * "resnet", num_layer)
        return (input_shape=input_shape, resnet=resnet)
    end
end
