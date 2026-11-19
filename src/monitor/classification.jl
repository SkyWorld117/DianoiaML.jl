module Classification
    function func(output::Array{Float32}, target::Array{Float32})
        size(output) == size(target) || throw(DimensionMismatch("prediction and target shapes differ"))
        predictions = reshape(output, size(output, 1), :)
        labels = reshape(target, size(target, 1), :)
        errors = 0
        for b in axes(predictions, 2)
            if size(predictions, 1) == 1
                errors += (predictions[1, b] >= 0.5f0) != (labels[1, b] >= 0.5f0)
            else
                @views errors += argmax(predictions[:, b]) != argmax(labels[:, b])
            end
        end
        return errors
    end
end
