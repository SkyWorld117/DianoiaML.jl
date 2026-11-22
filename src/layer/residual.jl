module ResidualM
    import ..DianoiaML: precision_type
    using ..DianoiaML: checked_precision, layer_input, model_precision
    import ..DianoiaML: backward!, supports_backward
    using ..resnet: ResNet
    using ..DianoiaML: AbstractLayer, AbstractLayerStack
    import ..DianoiaML
    import ..DianoiaML: initialize!, activate!, update!, save_layer, activate_layer, layer_children, validate_connections

    mutable struct Residual{T<:AbstractFloat, N,S<:AbstractLayerStack} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        resnet::S

        output::Array{T,N}
        δ::Array{T,N}

        function Residual(;input_shape::Tuple, resnet::AbstractLayerStack, precision::Type{T}=model_precision(resnet)) where {T<:AbstractFloat}
            checked_precision(T)
            new{T,length(input_shape) + 1, typeof(resnet)}(input_shape, input_shape, resnet)
        end
    end

    """
        Residual(network)

    Add a network's output to its input. The nested network must preserve the
    input shape.
    """
    function Residual(network::AbstractLayerStack; precision=model_precision(network))
        network.num_layer > 0 || throw(ArgumentError("a residual network requires at least one layer"))
        layer = Residual(; input_shape=network.input_shape, resnet=network, precision)
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

    precision_type(::Residual{T}) where {T} = T

    function initialize!(layer::Residual{T}, mini_batch::Int) where {T}
        initialize!(layer.resnet, mini_batch)
        validate_connections(layer)
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(T, layer.output_shape..., mini_batch)
        layer.δ = zeros(T, layer.input_shape..., mini_batch)
    end

    activate!(layer::Residual{T}, input::Array{T}; training::Bool=false) where {T} =
        activate_layer(layer, input, training)

    function activate_layer(layer::Residual, input, training)
        input = layer_input(layer, input)
        branch = activate_layer(layer.resnet, input, training)
        size(branch) == size(input) || throw(DimensionMismatch("the residual branch must preserve the input shape"))
        if !isdefined(layer, :output) || size(layer.output) != size(input)
            layer.output = similar(input)
            layer.δ = similar(input)
        end
        layer.output .= branch .+ input
    end

    supports_backward(layer::Residual) = supports_backward(layer.resnet)

    function backward!(layer::Residual{T}, input, delta::Array{T}) where {T}
        layer.δ .= delta .+ backward!(layer.resnet, layer_input(layer.resnet, input), delta)
    end

    function update!(layer::Residual{T}, optimizer::String, input::Array{T}, δₗ₊₁::Array{T}, α::Real, parameters::Tuple, direction::Int=1) where {T}
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
        DianoiaML.write_precision(file, id * "network_precision", layer.resnet)
        DianoiaML.save_stack_layers(file, layer.resnet, id * "resnet")
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        type_key = id * "network_type"
        name = haskey(file, type_key) ? read(file, type_key) : "ResNet"
        precision = DianoiaML.read_precision(file, id * "network_precision")
        resnet = DianoiaML.stack_from_name(name; precision)
        num_layer = read(file, id*"num_layer")
        DianoiaML.load_stack_layers!(resnet, file, id * "resnet", num_layer)
        return (input_shape=input_shape, resnet=resnet)
    end
end
