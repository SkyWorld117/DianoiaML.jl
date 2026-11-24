struct Absolute end
function (::Absolute)(output, target)
    check_loss_shape(output, target)
    A = promote_type(Float64, accumulation_type(eltype(output)), accumulation_type(eltype(target)))
    return sum(i -> abs(A(output[i]) - A(target[i])), eachindex(output, target))
end
