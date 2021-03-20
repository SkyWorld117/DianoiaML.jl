module upsampling2d
    using LoopVectorization

    mutable struct UpSampling2D
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_size::Int64
        input2D_size::Tuple{Int64, Int64}
        input_filter::Int64
        original_unit::Int64
        size::Tuple{Int64, Int64}
        layer_size::Int64
        activation_function::Module

        value::Array{Float32}
        output::Array{Float32}
        ∇biases::Array{Float32}
        propagation_units::Array{Float32}
        unit_size::Int64

        function UpSampling2D(;input_filter::Int64, input_size::Int64, input2D_size::Tuple{Int64, Int64}, size::Tuple{Int64, Int64}, activation_function::Module, reload::Bool=false)
            if reload
                return new(save_UpSampling2D, load_UpSampling2D, activate_UpSampling2D, init_UpSampling2D, update_UpSampling2D)
            end

            layer_size = size[1]*size[2]*input_size
            original_unit = input2D_size[1]*input2D_size[2]
            new(save_UpSampling2D, load_UpSampling2D, activate_UpSampling2D, init_UpSampling2D, update_UpSampling2D, input_size, input2D_size, input_filter, original_unit, size, layer_size, activation_function)
        end
    end

    function init_UpSampling2D(layer::UpSampling2D, mini_batch::Int64)
        layer.value = zeros(Float32, (layer.layer_size, mini_batch))
        layer.∇biases = zeros(Float32, (layer.layer_size, mini_batch))
        layer.propagation_units = zeros(Float32, (layer.input_size, mini_batch))
        layer.unit_size = layer.size[1]*layer.size[2]*layer.original_unit
    end

    function activate_UpSampling2D(layer::UpSampling2D, input::Array{Float32})
        upscale!(layer.value, input, layer.size, layer.input2D_size, layer.input_filter, layer.original_unit, layer.unit_size, layer.size[1]*layer.input2D_size[1])
        layer.output = layer.activation_function.func(layer.value)
    end

    function update_UpSampling2D(layer::UpSampling2D, optimizer::String, Last_Layer_output::Array{Float32}, Next_Layer_propagation_units::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=0)
        layer.activation_function.get_∇biases!(layer.∇biases, layer.value, Next_Layer_propagation_units)
        layer.propagation_units .= 0.0f0
        downscale!(layer.propagation_units, layer.∇biases, layer.size, layer.input2D_size, layer.input_filter, layer.original_unit, layer.unit_size, layer.size[1]*layer.input2D_size[1])
    end

    function upscale!(value::Array{Float32}, input::Array{Float32}, size::Tuple{Int64, Int64}, input2D_size::Tuple{Int64, Int64}, input_filter::Int64, original_unit::Int64, unit_size::Int64, new_input2D_size::Int64)
        @avx for f in 0:input_filter-1, s₁ in 1:size[1], s₂ in 0:size[2]-1, i in 0:input2D_size[1]-1, j in 0:input2D_size[2]-1, b in axes(value, 2)
            value[f*unit_size+(j*size[2]+s₂)*new_input2D_size+i*size[1]+s₁, b] = input[f*original_unit+j*input2D_size[1]+i+1, b]
        end
    end

    function downscale!(propagation_units::Array{Float32}, ∇biases::Array{Float32}, size::Tuple{Int64, Int64}, input2D_size::Tuple{Int64, Int64}, input_filter::Int64, original_unit::Int64, unit_size::Int64, new_input2D_size::Int64)
        @avx for f in 0:input_filter-1, s₁ in 1:size[1], s₂ in 0:size[2]-1, i in 0:input2D_size[1]-1, j in 0:input2D_size[2]-1, b in axes(∇biases, 2)
            propagation_units[f*original_unit+j*input2D_size[1]+i+1, b] += ∇biases[f*unit_size+(j*size[2]+s₂)*new_input2D_size+i*size[1]+s₁, b]/(size[1]*size[2])
        end
    end

    function save_UpSampling2D(layer::UpSampling2D, file::Any, id::Int64)
        write(file, string(id), "UpSampling2D")
        write(file, string(id)*"input_size", layer.input_size)
        write(file, string(id)*"input2D_size", collect(layer.input2D_size))
        write(file, string(id)*"size", collect(layer.size))
    end

    function load_UpSampling2D(layer::UpSampling2D, file::Any, id::Int64)
        layer.input_size = read(file, string(id)*"input_size")
        layer.input2D_size = Tuple(read(file, string(id)*"input2D_size"))
        layer.original_unit = layer.input2D_size[1]*layer.input2D_size[2]
        layer.input_filter = layer.input_size÷layer.original_unit
        layer.size = Tuple(read(file, string(id)*"size"))
        layer.layer_size = layer.size[1]*layer.size[2]*layer.input_size
    end
end
