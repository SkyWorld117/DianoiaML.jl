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

function loss_value(loss::Union{QuadraticLoss,MeanSquaredError}, ŷ, y)
    check_loss_shape(ŷ, y)
    gpu_array(ŷ) && return gpu_loss_value(loss, ŷ, y)
    A = loss_type(ŷ)
    return sum(i -> abs2(A(ŷ[i]) - A(y[i])), eachindex(ŷ, y)) * squared_scale(loss, ŷ)
end

function loss_gradient!(∇ŷ, loss::Union{QuadraticLoss,MeanSquaredError}, ŷ, y)
    check_loss_shape(ŷ, y)
    y = compatible_target(ŷ, y)
    A = loss_type(ŷ)
    scale = 2 * squared_scale(loss, ŷ)
    kernel_recording[] === nothing || return launch!(squared_gradient_gpu!, ∇ŷ, ŷ, y, scale)
    return ∇ŷ .= scale .* (A.(ŷ) .- A.(y))
end

function loss_value(::AbsoluteLoss, ŷ, y)
    check_loss_shape(ŷ, y)
    gpu_array(ŷ) && return gpu_loss_value(AbsoluteLoss(), ŷ, y)
    A = loss_type(ŷ)
    return sum(i -> abs(A(ŷ[i]) - A(y[i])), eachindex(ŷ, y))
end

function loss_gradient!(∇ŷ, ::AbsoluteLoss, ŷ, y)
    check_loss_shape(ŷ, y)
    y = compatible_target(ŷ, y)
    return broadcast_tensor!((prediction, target) -> sign(prediction - target), ∇ŷ, ŷ, y)
end

function loss_value(::CategoricalCrossEntropy, p, y)
    check_loss_shape(p, y)
    gpu_array(p) && return gpu_loss_value(CategoricalCrossEntropy(), p, y)
    ε = probability_floor(eltype(p))
    return sum(i -> -y[i] * log(max(p[i], ε)), eachindex(p, y))
end

function loss_gradient!(∇p, ::CategoricalCrossEntropy, p, y)
    check_loss_shape(p, y)
    y = compatible_target(p, y)
    return broadcast_tensor!((prediction, target, ε) -> -target / max(prediction, ε),
                             ∇p, p, y, probability_floor(eltype(p)))
end

function loss_value(::BinaryCrossEntropy, output, target)
    check_loss_shape(output, target)
    gpu_array(output) && return gpu_loss_value(BinaryCrossEntropy(), output, target)
    ε = probability_floor(eltype(output))
    return sum(eachindex(output, target)) do i
        y, p = target[i], output[i]
        -y * log(max(p, ε)) - (1 - y) * log(max(1 - p, ε))
    end
end

function loss_gradient!(∇p, ::BinaryCrossEntropy, p, y)
    check_loss_shape(p, y)
    y = compatible_target(p, y)
    ε = probability_floor(eltype(p))
    gradient(p, y, ε) = (1 - y) / max(1 - p, ε) - y / max(p, ε)
    return broadcast_tensor!(gradient, ∇p, p, y, ε)
end

function loss_value(::BinaryCrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    gpu_array(output) && return gpu_loss_value(BinaryCrossEntropyWithLogits(), output, target)
    A = loss_type(output)
    return sum(eachindex(output, target)) do i
        z, y = A(output[i]), A(target[i])
        max(z, zero(A)) - z * y + log1p(exp(-abs(z)))
    end
end

function loss_gradient!(∇z, ::BinaryCrossEntropyWithLogits, z, y)
    check_loss_shape(z, y)
    y = compatible_target(z, y)
    cast = ConvertTo{loss_type(z)}()
    return broadcast_tensor!((logit, target) -> sigmoid(cast(logit)) - cast(target), ∇z, z, y)
end

function loss_value(::CrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    gpu_array(output) && return gpu_loss_value(CrossEntropyWithLogits(), output, target)
    A = loss_type(output)
    width = size(output, 1)
    z, y = reshape(output, width, :), reshape(target, width, :)
    total = zero(A)
    @inbounds for col in axes(z, 2)
        peak = A(maximum(view(z, :, col)))
        log_Z = log(sum(i -> exp(A(z[i, col]) - peak), axes(z, 1)))
        @simd for i in axes(z, 1)
            total += A(y[i, col]) * (log_Z - (A(z[i, col]) - peak))
        end
    end
    return total
end

function loss_gradient!(dest, ::CrossEntropyWithLogits, output, target)
    check_loss_shape(output, target)
    gpu_array(output) && return gpu_crossentropy_gradient!(dest, output, target)
    A = loss_type(output)
    width = size(output, 1)
    z, y, ∇z = map(x -> reshape(x, width, :), (output, target, dest))
    @inbounds for col in axes(z, 2)
        peak = A(maximum(view(z, :, col)))
        Z, mass = zero(A), zero(A)
        @simd for i in axes(z, 1)
            Z += exp(A(z[i, col]) - peak)
            mass += A(y[i, col])
        end
        @simd for i in axes(z, 1)
            ∇z[i, col] = (mass / Z) * exp(A(z[i, col]) - peak) - A(y[i, col])
        end
    end
    return dest
end
