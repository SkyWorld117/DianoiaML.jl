module Classification
    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        size(output) == size(target) || throw(DimensionMismatch("prediction and target shapes differ"))
        predictions = reshape(output, size(output, 1), :)
        labels = reshape(target, size(target, 1), :)
        errors = 0
        for b in axes(predictions, 2)
            if size(predictions, 1) == 1
                errors += (predictions[1, b] >= 1//2) != (labels[1, b] >= 1//2)
            else
                @views errors += argmax(predictions[:, b]) != argmax(labels[:, b])
            end
        end
        return errors
    end
end
