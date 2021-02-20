module dense
    include("../tools/glorotuniform.jl")
    using .GlorotUniform
    using LoopVectorization

    mutable struct Dense
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_size::Int64
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
        propagation_units::Array{Float32}

        function Dense(;input_size::Int64, layer_size::Int64, activation_function::Module, randomization::Bool=true, reload::Bool=false)
            if reload
                return new(save_Dense, load_Dense, activate_Dense, init_Dense, update_Dense)
            end

            if randomization
                weights = Array{Float32}(GUMatrix(input_size, layer_size, (layer_size, input_size)))
                biases = Array{Float32}(GUMatrix(input_size, layer_size, layer_size))
            else
                weights = zeros(Float32, (layer_size, input_size))
                biases = zeros(Float32, layer_size)
            end

            Vdw = zeros(Float32, size(weights))
            Sdw = zeros(Float32, size(weights))
            Vdb = zeros(Float32, size(biases))
            Sdb = zeros(Float32, size(biases))
            new(save_Dense, load_Dense, activate_Dense, init_Dense, update_Dense, input_size, layer_size, activation_function, weights, biases, Vdw, Sdw, Vdb, Sdb)
        end
    end

    function init_Dense(layer::Dense, mini_batch::Int64)
        layer.value = zeros(Float32, (size(layer.weights, 1), mini_batch))
        layer.propagation_units = zeros(Float32, (size(layer.weights, 2), mini_batch))
    end

    function activate_Dense(layer::Dense, input::Array{Float32})
        @avx for x in axes(layer.weights, 1), y in axes(input, 2)
            c = 0.0f0
            for z in axes(layer.weights, 2)
                c += layer.weights[x,z]*input[z,y]
            end
            layer.value[x,y] = c+layer.biases[x]
        end
        # layer.value = layer.weights*input .+ layer.biases
        layer.output = layer.activation_function.func(layer.value)
    end

    function update_Dense(layer::Dense, optimizer::String, Last_Layer_output::Array{Float32}, Next_Layer_propagation_units::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        ∇biases = layer.activation_function.get_∇biases(layer.value, Next_Layer_propagation_units)
        @avx for x in axes(layer.weights, 2), y in axes(∇biases, 2)
            c = 0.0f0
            for z in axes(layer.weights, 1)
                c += layer.weights[z,x]*∇biases[z,y]
            end
            layer.propagation_units[x,y] = c
        end
        # layer.propagation_units = transpose(layer.weights)*∇biases

        if optimizer=="SGD"
            @avx for x in axes(∇biases, 1), y in axes(Last_Layer_output, 1)
                layer.weights[x,y] -= α*∇biases[x,1]*Last_Layer_output[y,1]*direction
            end
            # layer.weights -= ∇biases*transpose(Last_Layer_output).*α
            @avx for i in 1:length(layer.biases)
                layer.biases[i] -= α*∇biases[i,1]*direction
            end
            # layer.biases -= sum(∇biases, dims=2).*α

        elseif optimizer=="Minibatch_GD"
            @avx for x in axes(∇biases, 1), y in axes(Last_Layer_output, 1)
                c = 0.0f0
                for b in axes(∇biases, 2)
                    c += ∇biases[x,b]*Last_Layer_output[y,b]
                end
                layer.weights[x,y] -= c*α*direction
            end
            # layer.weights -= ∇biases*transpose(Last_Layer_output).*α
            @avx for i in 1:length(layer.biases)
                c = 0.0f0
                for b in axes(∇biases, 2)
                    c += ∇biases[i,b]
                end
                layer.biases[i] -= c*α*direction
            end
            # layer.biases -= sum(∇biases, dims=2).*α

        elseif optimizer=="Adam"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]

            @avx for x in axes(∇biases, 1), y in axes(Last_Layer_output, 1)
                layer.Vdw[x,y] = layer.Vdw[x,y]*β₁ + ∇biases[x,1]*Last_Layer_output[y,1]*(1-β₁)
                layer.Sdw[x,y] = layer.Sdw[x,y]*β₂ + (∇biases[x,1]*Last_Layer_output[y,1])^2*(1-β₂)
                layer.weights[x,y] -= α*(layer.Vdw[x,y]/(1-β₁^t))/(sqrt(layer.Sdw[x,y]/(1-β₂^t))+ϵ)*direction
            end

            @avx for i in 1:length(layer.biases)
                layer.Vdb[i] = layer.Vdb[i]*β₁ + ∇biases[i,1]*(1-β₁)
                layer.Sdb[i] = layer.Sdb[i]*β₂ + ∇biases[i,1]^2*(1-β₂)
                layer.biases[i] -= α*(layer.Vdb[i]/(1-β₁^t))/(sqrt(layer.Sdb[i]/(1-β₂^t))+ϵ)*direction
            end

        elseif optimizer=="AdaBelief"
            t = parameters[1]
            β₁ = parameters[2]
            β₂ = parameters[3]
            ϵ = parameters[4]

            @avx for x in axes(∇biases, 1), y in axes(Last_Layer_output, 1)
                layer.Vdw[x,y] = layer.Vdw[x,y]*β₁ + ∇biases[x,1]*Last_Layer_output[y,1]*(1-β₁)
                layer.Sdw[x,y] = layer.Sdw[x,y]*β₂ + (∇biases[x,1]*Last_Layer_output[y,1]-layer.Vdw[x,y])^2*(1-β₂) + ϵ
                layer.weights[x,y] -= α*(layer.Vdw[x,y]/(1-β₁^t))/(sqrt(layer.Sdw[x,y]/(1-β₂^t))+ϵ)*direction
            end

            @avx for i in 1:length(layer.biases)
                layer.Vdb[i] = layer.Vdb[i]*β₁ + ∇biases[i,1]*(1-β₁)
                layer.Sdb[i] = layer.Sdb[i]*β₂ + (∇biases[i,1]-layer.Vdb[i])^2*(1-β₂)
                layer.biases[i] -= α*(layer.Vdb[i]/(1-β₁^t))/(sqrt(layer.Sdb[i]/(1-β₂^t))+ϵ)*direction
            end
        end
    end

    function save_Dense(layer::Dense, file::Any, id::Int64)
        write(file, string(id), "Dense")
        write(file, string(id)*"weights", layer.weights)
        write(file, string(id)*"biases", layer.biases)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_Dense(layer::Dense, file::Any, id::Int64)
        layer.weights = read(file, string(id)*"weights")
        layer.biases = read(file, string(id)*"biases")
        layer.input_size = size(layer.weights, 2)
        layer.layer_size = size(layer.weights, 1)
        layer.Vdw = zeros(Float32, size(layer.weights))
        layer.Sdw = zeros(Float32, size(layer.weights))
        layer.Vdb = zeros(Float32, size(layer.biases))
        layer.Sdb = zeros(Float32, size(layer.biases))
    end
end
