struct Classification{T<:Real}
    threshold::T
end
Classification(; threshold=1//2) = Classification(threshold)

function (monitor::Classification)(output, target)
    check_loss_shape(output, target)
    width = size(output, 1)
    predicted, expected = reshape(output, width, :), reshape(target, width, :)
    errors = 0
    for col in axes(predicted, 2)
        if width == 1
            errors += (predicted[1, col] >= monitor.threshold) != (expected[1, col] >= 1//2)
        else
            @views errors += argmax(predicted[:, col]) != argmax(expected[:, col])
        end
    end
    return errors
end
