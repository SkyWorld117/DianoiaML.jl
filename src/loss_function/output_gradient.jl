function output_gradient!(delta, layer, target, loss)
    output = layer.output
    size(delta) == size(output) == size(target) || throw(DimensionMismatch("prediction and target shapes differ"))
    activation = layer isa Union{Dense,Conv2D,TConv2D,UpSampling2D} ? layer.activation_function : nothing
    if loss === Categorical_Cross_Entropy_Loss && (activation === Softmax || activation === Softmax_CEL)
        # d/dlogits of cross-entropy also covers weighted and smoothed targets.
        classes = size(output, 1)
        for first in 1:classes:length(output)
            last = first + classes - 1
            mass = 0f0
            @inbounds @simd for i in first:last
                mass += target[i]
            end
            @inbounds @simd for i in first:last
                delta[i] = output[i] * mass - target[i]
            end
        end
        return LogitGradient(delta)
    elseif loss === Binary_Cross_Entropy_Loss && activation === Sigmoid
        delta .= output .- target
        return LogitGradient(delta)
    end

    loss.prop!(delta, output, target)
    return delta
end
