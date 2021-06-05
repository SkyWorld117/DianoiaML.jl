module DenseM
    include("../tools/glorotuniform.jl")
    using .GlorotUniform
    using LoopVectorization

    mutable struct Dense
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple
        layer_size::Int64
        activation_function::Module

        weights::Array{Float32}
        biases::Array{Float32}

        Vdw::Array{Float32}
        Sdw::Array{Float32}
        Vdb::Array{Float32}
        Sdb::Array{Float32}

        value::Array{Float32}
        output::Array{Float32}
        ∇biases::Array{Float32}
        δ::Array{Float32}

        function Dense(;input_shape::Tuple, layer_size::Int64, activation_function::Module, randomization::Bool=true)
            if randomization
                weights = Array{Float32}(GUMatrix(input_shape[1], layer_size, (layer_size, input_shape[1])))
                biases = Array{Float32}(GUMatrix(input_shape[1], layer_size, layer_size))
            else
                weights = zeros(Float32, (layer_size, input_shape[1]))
                biases = zeros(Float32, layer_size)
            end

            output_shape = (layer_size,)

            Vdw = zeros(Float32, size(weights))
            Sdw = zeros(Float32, size(weights))
            Vdb = zeros(Float32, size(biases))
            Sdb = zeros(Float32, size(biases))
            new(save_Dense, load_Dense, activate_Dense, init_Dense, update_Dense, input_shape, output_shape, layer_size, activation_function, weights, biases, Vdw, Sdw, Vdb, Sdb)
        end
    end

    function init_Dense(layer::Dense, mini_batch::Int64)
        layer.value = zeros(Float32, (layer.layer_size, mini_batch))
        layer.output = zeros(Float32, (layer.layer_size, mini_batch))
        layer.∇biases = zeros(Float32, (layer.layer_size, mini_batch))
        layer.δ = zeros(Float32, (layer.input_shape[1], mini_batch))
    end

    function activate_Dense(layer::Dense, input::Array{Float32})
        @avxt for x in axes(layer.weights, 1), y in axes(input, 2)
            c = 0.0f0
            for z in axes(layer.weights, 2)
                c += layer.weights[x,z]*input[z,y]
            end
            layer.value[x,y] = c+layer.biases[x]
        end
        # layer.value = layer.weights*input .+ layer.biases
        layer.activation_function.func!(layer.output, layer.value)
    end

    function update_Dense(layer::Dense, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        layer.activation_function.get_∇biases!(layer.∇biases, layer.value, δₗ₊₁)
        @avxt for x in axes(layer.weights, 2), y in axes(layer.∇biases, 2)
            c = 0.0f0
            for z in axes(layer.weights, 1)
                c += layer.weights[z,x]*layer.∇biases[z,y]
            end
            layer.δ[x,y] = c
        end
        # layer.δ = transpose(layer.weights)*∇biases

        if optimizer=="SGD"
            @avxt for x in axes(layer.∇biases, 1), y in axes(input, 1)
                layer.weights[x,y] -= α*layer.∇biases[x,1]*input[y,1]*direction
            end
            # layer.weights -= ∇biases*transpose(input).*α
            @avxt for i in 1:length(layer.biases)
                layer.biases[i] -= α*layer.∇biases[i,1]*direction
            end
            # layer.biases -= sum(∇biases, dims=2).*α

        elseif optimizer=="Minibatch_GD"
            batch_size = size(input, 2)
            @avxt for x in axes(layer.∇biases, 1), y in axes(input, 1)
                c = 0.0f0
                for b in axes(layer.∇biases, 2)
                    c += layer.∇biases[x,b]*input[y,b]
                end
                layer.weights[x,y] -= c*α*direction/batch_size
            end
            # layer.weights -= ∇biases*transpose(input).*α
            @avxt for i in 1:length(layer.biases)
                c = 0.0f0
                for b in axes(layer.∇biases, 2)
                    c += layer.∇biases[i,b]
                end
                layer.biases[i] -= c*α*direction/batch_size
            end
            # layer.biases -= sum(∇biases, dims=2).*α

        elseif optimizer=="Adam"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]

            @avxt for x in axes(layer.∇biases, 1), y in axes(input, 1)
                layer.Vdw[x,y] = layer.Vdw[x,y]*β₁ + layer.∇biases[x,1]*input[y,1]*(1-β₁)
                layer.Sdw[x,y] = layer.Sdw[x,y]*β₂ + (layer.∇biases[x,1]*input[y,1])^2*(1-β₂)
                layer.weights[x,y] -= α*(layer.Vdw[x,y]/(1-β₁^t))/(sqrt(layer.Sdw[x,y]/(1-β₂^t))+ϵ)*direction
            end

            @avxt for i in 1:length(layer.biases)
                layer.Vdb[i] = layer.Vdb[i]*β₁ + layer.∇biases[i,1]*(1-β₁)
                layer.Sdb[i] = layer.Sdb[i]*β₂ + layer.∇biases[i,1]^2*(1-β₂)
                layer.biases[i] -= α*(layer.Vdb[i]/(1-β₁^t))/(sqrt(layer.Sdb[i]/(1-β₂^t))+ϵ)*direction
            end

        elseif optimizer=="AdaBelief"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]

            @avxt for x in axes(layer.∇biases, 1), y in axes(input, 1)
                layer.Vdw[x,y] = layer.Vdw[x,y]*β₁ + layer.∇biases[x,1]*input[y,1]*(1-β₁)
                layer.Sdw[x,y] = layer.Sdw[x,y]*β₂ + (layer.∇biases[x,1]*input[y,1]-layer.Vdw[x,y])^2*(1-β₂) + ϵ
                layer.weights[x,y] -= α*(layer.Vdw[x,y]/(1-β₁^t))/(sqrt(layer.Sdw[x,y]/(1-β₂^t))+ϵ)*direction
            end

            @avxt for i in 1:length(layer.biases)
                layer.Vdb[i] = layer.Vdb[i]*β₁ + layer.∇biases[i,1]*(1-β₁)
                layer.Sdb[i] = layer.Sdb[i]*β₂ + (layer.∇biases[i,1]-layer.Vdb[i])^2*(1-β₂)
                layer.biases[i] -= α*(layer.Vdb[i]/(1-β₁^t))/(sqrt(layer.Sdb[i]/(1-β₂^t))+ϵ)*direction
            end
        end
    end

    function save_Dense(layer::Dense, file::Any, id::String)
        write(file, id, "Dense")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"layer_size", layer.layer_size)
        write(file, id*"activation_function", layer.activation_function.get_name())

        write(file, id*"weights", layer.weights)
        write(file, id*"biases", layer.biases)
    end

    function load_Dense(layer::Dense, file::Any, id::String)
        layer.weights = read(file, id*"weights")
        layer.biases = read(file, id*"biases")
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        layer_size = read(file, id*"layer_size")
        return (input_shape=input_shape, layer_size=layer_size)
    end
end
