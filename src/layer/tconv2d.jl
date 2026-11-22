module TConv2DM
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, load_layer!
    using ..GlorotUniform: GUMatrix
    using ..DianoiaML: activation_gradient!
    using ..SpatialConvolution: Workspace, forward!, backward!

    mutable struct TConv2D <: AbstractLayer
        input_shape::NTuple{3,Int}
        output_shape::NTuple{3,Int}
        kernel_size::NTuple{2,Int}
        activation_function::Module

        filters::Array{Float32,4}
        biases::Vector{Float32}

        padding::Int
        filter::Int
        strides::NTuple{2,Int}

        ∇weights::Array{Float32,4}
        Vdw::Array{Float32,4}
        Sdw::Array{Float32,4}
        ∇biases_c::Vector{Float32}
        Vdb::Vector{Float32}
        Sdb::Vector{Float32}

        value::Array{Float32,4}
        output::Array{Float32,4}
        ∇biases::Array{Float32,4}
        δ::Array{Float32,4}
        workspace::Workspace
        padding_input::Array{Float32,4}
        padded_gradient::Array{Float32,4}

        function TConv2D(;input_shape::Tuple, filter::Int, padding::Int=0, kernel_size::NTuple{2,Int},
                          strides::NTuple{2,Int}=(1, 1), activation_function::Module, randomization::Bool=true)
            length(input_shape) == 3 && all(>(0), input_shape) ||
                throw(ArgumentError("TConv2D requires positive spatial dimensions and channels"))
            padding >= 0 && filter > 0 && all(>(0), strides) && all(>(0), kernel_size) ||
                throw(ArgumentError("invalid TConv2D padding, filter count, stride or kernel size"))
            # Keep the zero-insertion geometry used by existing models.
            height, width = ntuple(d -> (input_shape[d] - 1) * strides[d] + 2 * padding - kernel_size[d] + 2, 2)
            height > 0 && width > 0 || throw(ArgumentError("TConv2D output dimensions must be positive"))
            output_shape = (height, width, filter)
            filter_size = (filter, input_shape[3], kernel_size...)
            area = prod(kernel_size)
            filters = randomization ? GUMatrix(area * input_shape[3], area * filter, filter_size) : zeros(Float32, filter_size)
            biases = zeros(Float32, filter)

            new(input_shape, output_shape, kernel_size, activation_function, filters, biases, padding, filter,
                strides, similar(filters), zeros(Float32, filter_size), zeros(Float32, filter_size),
                similar(biases), zeros(Float32, filter), zeros(Float32, filter))
        end
    end

    function initialize!(layer::TConv2D, mini_batch::Int)
        isdefined(layer, :value) && size(layer.value, 4) == mini_batch && return
        layer.value = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = similar(layer.value)
        layer.∇biases = similar(layer.value)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
        expanded = ntuple(d -> (layer.input_shape[d] - 1) * layer.strides[d] + 2 * layer.padding + 1, 2)
        layer.padding_input = zeros(Float32, expanded..., layer.input_shape[3], mini_batch)
        layer.padded_gradient = similar(layer.padding_input)
        layer.workspace = Workspace(size(layer.padding_input), layer.filters; stride=(1, 1), padding=0)
    end

    function activate!(layer::TConv2D, input::Array{Float32,4})
        size(input) == size(layer.δ) || throw(DimensionMismatch("unexpected transposed convolution input size"))
        x, y = layer.strides
        p = layer.padding
        @views layer.padding_input[p + 1:x:p + 1 + (size(input, 1) - 1) * x,
                                  p + 1:y:p + 1 + (size(input, 2) - 1) * y, :, :] .= input
        forward!(layer, layer.padding_input)
    end

    function update!(layer::TConv2D, optimizer::String, input::Array{Float32,4}, δₗ₊₁,
                              α::Float64, parameters::Tuple, direction::Int=1)
        activation_gradient!(layer.∇biases, layer.activation_function, layer.value, δₗ₊₁)
        backward!(layer, layer.padding_input, layer.padded_gradient, optimizer, α, parameters, direction)
        x, y = layer.strides
        p = layer.padding
        @views layer.δ .= layer.padded_gradient[p + 1:x:p + 1 + (size(input, 1) - 1) * x,
                                               p + 1:y:p + 1 + (size(input, 2) - 1) * y, :, :]
    end

    function save_layer(layer::TConv2D, file, id::String)
        write(file, id, "TConv2D")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"filter", layer.filter)
        write(file, id*"padding", layer.padding)
        write(file, id*"kernel_size", collect(layer.kernel_size))
        write(file, id*"strides", collect(layer.strides))
        write(file, id*"activation_function", layer.activation_function.get_name())

        write(file, id*"filters", layer.filters)
        write(file, id*"biases", layer.biases)
    end

    function load_layer!(layer::TConv2D, file, id::String)
        layer.filters = read(file, id*"filters")
        layer.biases = read(file, id*"biases")
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
