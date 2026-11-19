module Conv2DM
    using ..GlorotUniform: GUMatrix
    using ..DianoiaML: activation_gradient!
    using ..SpatialConvolution: Workspace, forward!, backward!

    mutable struct Conv2D
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

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

        function Conv2D(;input_shape::Tuple, filter::Int, padding::Int=0, kernel_size::NTuple{2,Int},
                          strides::NTuple{2,Int}=(1, 1), activation_function::Module, randomization::Bool=true)
            length(input_shape) == 3 && all(>(0), input_shape) ||
                throw(ArgumentError("Conv2D requires positive spatial dimensions and channels"))
            padding >= 0 && filter > 0 && all(>(0), strides) && all(>(0), kernel_size) ||
                throw(ArgumentError("invalid Conv2D padding, filter count, stride or kernel size"))
            height, width = ntuple(d -> fld(input_shape[d] + 2 * padding - kernel_size[d], strides[d]) + 1, 2)
            height > 0 && width > 0 || throw(ArgumentError("Conv2D output dimensions must be positive"))
            output_shape = (height, width, filter)
            filter_size = (filter, input_shape[3], kernel_size...)
            area = prod(kernel_size)
            filters = randomization ? GUMatrix(area * input_shape[3], area * filter, filter_size) : zeros(Float32, filter_size)
            biases = zeros(Float32, filter)

            new(save_Conv2D, load_Conv2D, activate_Conv2D, init_Conv2D, update_Conv2D,
                input_shape, output_shape, kernel_size, activation_function, filters, biases, padding, filter,
                strides, similar(filters), zeros(Float32, filter_size), zeros(Float32, filter_size),
                similar(biases), zeros(Float32, filter), zeros(Float32, filter))
        end
    end

    function init_Conv2D(layer::Conv2D, mini_batch::Int)
        isdefined(layer, :value) && size(layer.value, 4) == mini_batch && return
        layer.value = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = similar(layer.value)
        layer.∇biases = similar(layer.value)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
        layer.workspace = Workspace(size(layer.δ), layer.filters; stride=layer.strides, padding=layer.padding)
    end

    function activate_Conv2D(layer::Conv2D, input::Array{Float32,4})
        forward!(layer, input)
    end

    function update_Conv2D(layer::Conv2D, optimizer::String, input::Array{Float32,4}, δₗ₊₁,
                              α::Float64, parameters::Tuple, direction::Int=1)
        activation_gradient!(layer.∇biases, layer.activation_function, layer.value, δₗ₊₁)
        backward!(layer, input, layer.δ, optimizer, α, parameters, direction)
    end

    function save_Conv2D(layer::Conv2D, file::Any, id::String)
        write(file, id, "Conv2D")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"filter", layer.filter)
        write(file, id*"padding", layer.padding)
        write(file, id*"kernel_size", collect(layer.kernel_size))
        write(file, id*"strides", collect(layer.strides))
        write(file, id*"activation_function", layer.activation_function.get_name())

        write(file, id*"filters", layer.filters)
        write(file, id*"biases", layer.biases)
    end

    function load_Conv2D(layer::Conv2D, file::Any, id::String)
        layer.filters = read(file, id*"filters")
        layer.biases = read(file, id*"biases")
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        filter = read(file, id*"filter")
        padding = read(file, id*"padding")
        kernel_size = tuple(read(file, id*"kernel_size")...)
        strides = tuple(read(file, id*"strides")...)
        return (input_shape=input_shape, filter=filter, padding=padding, kernel_size=kernel_size, strides=strides)
    end
end
