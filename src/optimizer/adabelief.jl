module AdaBelief
    using ..DianoiaML: fit_gradient!
    using Random: default_rng

    function fit(;model, input_data::Array{Float32}, output_data::Array{Float32}, loss_function, monitor,
                 α::Float64=0.001, epochs::Int=20, batch::Real=32, β₁::Float64=0.9, β₂::Float64=0.999, ϵ::Float64=1e-8,
                 shuffle::Bool=true, rng=default_rng(), verbose::Bool=true)
        return fit_gradient!(model, input_data, output_data, loss_function, monitor, "AdaBelief", α,
                             epochs, batch, (β₁, β₂, ϵ); shuffle, rng, verbose)
    end
end
