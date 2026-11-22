module Adam
    using ..DianoiaML: fit_gradient!
    using Random: default_rng

    function fit(;model, input_data::AbstractArray{<:AbstractFloat}, output_data::AbstractArray{<:AbstractFloat}, loss_function, monitor,
                 α::Real=0.001, epochs::Int=20, batch::Real=32, β₁::Real=0.9, β₂::Real=0.999, ϵ::Real=1e-8,
                 shuffle::Bool=true, rng=default_rng(), verbose::Bool=true)
        return fit_gradient!(model, input_data, output_data, loss_function, monitor, "Adam", α,
                             epochs, batch, (β₁, β₂, ϵ); shuffle, rng, verbose)
    end
end
