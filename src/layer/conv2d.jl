module Conv2DM
    import ..DianoiaML: precision_type
    using ..DianoiaML: write_array, read_array
    using ..DianoiaML: checked_precision, accumulation_type
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, load_layer!
    using ..GlorotUniform: GUMatrix
    using ..DianoiaML: activation_gradient!
    using ..SpatialConvolution: Workspace, forward!
    import ..SpatialConvolution
    import ..DianoiaML: backward!, supports_backward, parameter_gradients
    using ..DianoiaML: update_parameters!

    mutable struct Conv2D{T<:AbstractFloat, A<:AbstractFloat} <: AbstractLayer
        input_shape::NTuple{3,Int}
        output_shape::NTuple{3,Int}
        kernel_size::NTuple{2,Int}
        activation_function::Module

        filters::Array{T,4}
        biases::Vector{T}

        padding::Int
        filter::Int
        strides::NTuple{2,Int}

        ∇weights::Array{A,4}
        Vdw::Array{A,4}
        Sdw::Array{A,4}
        ∇biases_c::Vector{A}
        Vdb::Vector{A}
        Sdb::Vector{A}

        value::Array{T,4}
        output::Array{T,4}
        ∇biases::Array{T,4}
        δ::Array{T,4}
        workspace::Workspace

        function Conv2D(;input_shape::Tuple, filter::Int, padding::Int=0, kernel_size::NTuple{2,Int},
                          strides::NTuple{2,Int}=(1, 1), activation_function::Module, randomization::Bool=true, precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            A = accumulation_type(T)
            length(input_shape) == 3 && all(>(0), input_shape) ||
                throw(ArgumentError("Conv2D requires positive spatial dimensions and channels"))
            padding >= 0 && filter > 0 && all(>(0), strides) && all(>(0), kernel_size) ||
                throw(ArgumentError("invalid Conv2D padding, filter count, stride or kernel size"))
            height, width = ntuple(d -> fld(input_shape[d] + 2 * padding - kernel_size[d], strides[d]) + 1, 2)
            height > 0 && width > 0 || throw(ArgumentError("Conv2D output dimensions must be positive"))
            output_shape = (height, width, filter)
            filter_size = (filter, input_shape[3], kernel_size...)
            area = prod(kernel_size)
            filters = randomization ? GUMatrix(area * input_shape[3], area * filter, filter_size, T) : zeros(T, filter_size)
            biases = zeros(T, filter)

            new{T,A}(input_shape, output_shape, kernel_size, activation_function, filters, biases, padding, filter,
                strides, similar(filters, A), zeros(A, filter_size), zeros(A, filter_size),
                similar(biases, A), zeros(A, filter), zeros(A, filter))
        end
    end

    precision_type(::Conv2D{T}) where {T} = T

    function initialize!(layer::Conv2D{T}, mini_batch::Int) where {T}
        isdefined(layer, :value) && size(layer.value, 4) == mini_batch && return
        layer.value = zeros(T, layer.output_shape..., mini_batch)
        layer.output = similar(layer.value)
        layer.∇biases = similar(layer.value)
        layer.δ = zeros(T, layer.input_shape..., mini_batch)
        layer.workspace = Workspace(size(layer.δ), layer.filters; stride=layer.strides, padding=layer.padding)
    end

    function activate!(layer::Conv2D{T}, input::Array{T,4}) where {T}
        forward!(layer, input)
    end

    supports_backward(::Conv2D) = true
    parameter_gradients(layer::Conv2D) =
        (; filters=(layer.filters, layer.∇weights), biases=(layer.biases, layer.∇biases_c))

    function backward!(layer::Conv2D{T}, input::Array{T,4}, δₗ₊₁; compute_parameters=true) where {T}
        activation_gradient!(layer.∇biases, layer.activation_function, layer.value, δₗ₊₁)
        SpatialConvolution.backward!(layer, input, layer.δ; compute_parameters)
        return layer.δ
    end

    function update!(layer::Conv2D{T}, optimizer::String, input::Array{T,4}, delta,
                     α::Real, parameters::Tuple, direction::Int=1) where {T}
        backward!(layer, input, delta; compute_parameters=direction != 0)
        direction == 0 && return
        rate = accumulation_type(T)(α * direction)
        update_parameters!(optimizer, layer.filters, layer.∇weights, layer.Vdw, layer.Sdw, rate, parameters)
        update_parameters!(optimizer, layer.biases, layer.∇biases_c, layer.Vdb, layer.Sdb, rate, parameters)
    end

    function save_layer(layer::Conv2D, file, id::String)
        write(file, id, "Conv2D")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"filter", layer.filter)
        write(file, id*"padding", layer.padding)
        write(file, id*"kernel_size", collect(layer.kernel_size))
        write(file, id*"strides", collect(layer.strides))
        write(file, id*"activation_function", layer.activation_function.get_name())

        write_array(file, id*"filters", layer.filters)
        write_array(file, id*"biases", layer.biases)
    end

    function load_layer!(layer::Conv2D, file, id::String)
        layer.filters = read_array(file, id*"filters")
        layer.biases = read_array(file, id*"biases")
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        filter = read(file, id*"filter")
        padding = read(file, id*"padding")
        kernel_size = tuple(read(file, id*"kernel_size")...)
        strides = tuple(read(file, id*"strides")...)
        return (input_shape=input_shape, filter=filter, padding=padding, kernel_size=kernel_size, strides=strides)
    end
end
