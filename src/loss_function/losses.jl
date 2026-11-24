abstract type AbstractLoss end

struct MeanSquaredError <: AbstractLoss end
struct QuadraticLoss <: AbstractLoss end
struct AbsoluteLoss <: AbstractLoss end
struct CategoricalCrossEntropy <: AbstractLoss end
struct BinaryCrossEntropy <: AbstractLoss end
struct CrossEntropyWithLogits <: AbstractLoss end
struct BinaryCrossEntropyWithLogits <: AbstractLoss end

loss_type(output) = accumulation_type(eltype(output))
feature_count(output) = ndims(output) == 1 ? length(output) : prod(Base.front(size(output)))
(loss::AbstractLoss)(output, target) = loss_value(loss, output, target)

function check_loss_shape(output, target)
    size(output) == size(target) || throw(DimensionMismatch("prediction and target shapes differ"))
end

squared_scale(::QuadraticLoss, output) = one(loss_type(output))
squared_scale(::MeanSquaredError, output) = inv(loss_type(output)(feature_count(output)))

function loss_value(loss::Union{QuadraticLoss,MeanSquaredError}, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    return sum(i -> abs2(A(output[i]) - A(target[i])), eachindex(output, target)) * squared_scale(loss, output)
end

function loss_gradient!(dest, loss::Union{QuadraticLoss,MeanSquaredError}, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    scale = 2 * squared_scale(loss, output)
    return dest .= scale .* (A.(output) .- A.(target))
end

function loss_value(::AbsoluteLoss, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    return sum(i -> abs(A(output[i]) - A(target[i])), eachindex(output, target))
end

function loss_gradient!(dest, ::AbsoluteLoss, output, target)
    check_loss_shape(output, target)
    return dest .= sign.(output .- target)
end

function loss_value(::CategoricalCrossEntropy, output, target)
    check_loss_shape(output, target)
    floor = probability_floor(eltype(output))
    return sum(i -> -target[i] * log(max(output[i], floor)), eachindex(output, target))
end

function loss_gradient!(dest, ::CategoricalCrossEntropy, output, target)
    check_loss_shape(output, target)
    return dest .= -target ./ max.(output, probability_floor(eltype(output)))
end

function loss_value(::BinaryCrossEntropy, output, target)
    check_loss_shape(output, target)
    floor = probability_floor(eltype(output))
    return sum(eachindex(output, target)) do i
        y, p = target[i], output[i]
        -y * log(max(p, floor)) - (1 - y) * log(max(1 - p, floor))
    end
end

function loss_gradient!(dest, ::BinaryCrossEntropy, output, target)
    check_loss_shape(output, target)
    floor = probability_floor(eltype(output))
    return dest .= (1 .- target) ./ max.(1 .- output, floor) .- target ./ max.(output, floor)
end

function loss_value(::BinaryCrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    return sum(eachindex(output, target)) do i
        x, y = A(output[i]), A(target[i])
        max(x, zero(A)) - x * y + log1p(exp(-abs(x)))
    end
end

function loss_gradient!(dest, ::BinaryCrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    return dest .= sigmoid.(A.(output)) .- A.(target)
end

function loss_value(::CrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    width = size(output, 1)
    logits, labels = reshape(output, width, :), reshape(target, width, :)
    total = zero(A)
    @inbounds for col in axes(logits, 2)
        peak = A(maximum(view(logits, :, col)))
        normalizer = log(sum(i -> exp(A(logits[i, col]) - peak), axes(logits, 1)))
        @simd for i in axes(logits, 1)
            total += A(labels[i, col]) * (normalizer - (A(logits[i, col]) - peak))
        end
    end
    return total
end

function loss_gradient!(dest, ::CrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    A = loss_type(output)
    width = size(output, 1)
    logits, labels, gradient = map(x -> reshape(x, width, :), (output, target, dest))
    @inbounds for col in axes(logits, 2)
        peak = A(maximum(view(logits, :, col)))
        normalizer, mass = zero(A), zero(A)
        @simd for i in axes(logits, 1)
            normalizer += exp(A(logits[i, col]) - peak)
            mass += A(labels[i, col])
        end
        @simd for i in axes(logits, 1)
            gradient[i, col] = (mass / normalizer) * exp(A(logits[i, col]) - peak) - A(labels[i, col])
        end
    end
    return dest
end
