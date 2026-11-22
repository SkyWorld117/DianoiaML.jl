module SGD
    using ..DianoiaML: fit_gradient!
    using Random: default_rng

    """
        SGD.fit(; model, input_data, output_data, loss_function, monitor, batch=1, ...)

    Visit every example once per epoch. Use `batch=1` for stochastic updates,
    a larger batch for minibatches, or the number of examples for full-batch
    gradient descent. Each update averages the gradients in its batch.
    """
    function fit(;model, input_data::AbstractArray{<:AbstractFloat}, output_data::AbstractArray{<:AbstractFloat}, loss_function, monitor,
                 α::Real=0.01, epochs::Int=20, batch::Real=1,
                 shuffle::Bool=true, rng=default_rng(), verbose::Bool=true)
        return fit_gradient!(model, input_data, output_data, loss_function, monitor, "SGD", α,
                             epochs, batch, (); shuffle, rng, verbose)
    end
end
