module Minibatch_GD
    using ..SGD

    function fit(; batch::Real=32, mini_batch::Union{Int,Nothing}=nothing, kwargs...)
        Base.depwarn("Use SGD.fit(; batch=...) for minibatch gradient descent.", :fit)
        return SGD.fit(; batch=something(mini_batch, batch), kwargs...)
    end
end
