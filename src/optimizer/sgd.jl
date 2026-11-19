module SGD
    using ..DianoiaML: fit_gradient!
    using Random: default_rng

    function fit(;model, input_data::Array{Float32}, output_data::Array{Float32}, loss_function, monitor,
                 α::Float64=0.01, epochs::Int=20, batch::Real=1,
                 shuffle::Bool=true, rng=default_rng(), verbose::Bool=true)
        return fit_gradient!(model, input_data, output_data, loss_function, monitor, "SGD", α,
                             epochs, batch, (); shuffle, rng, verbose)
    end
end
