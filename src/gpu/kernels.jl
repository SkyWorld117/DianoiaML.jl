@kernel inbounds=true function gather_examples_gpu!(output, input, indices, sample_width)
    i = @index(Global, Linear)
    sample, offset = divrem(i - 1, sample_width)
    output[i] = input[(Int(indices[sample + 1]) - 1) * sample_width + offset + 1]
end

function gather_examples(input, indices)
    output = similar(input, eltype(input), (Base.front(size(input))..., length(indices)))
    device_indices = indices isa UnitRange ? indices : device_copy(device_of(input), collect(indices))
    return launch!(gather_examples_gpu!, output, input, device_indices, prod(Base.front(size(input))))
end

# Shared tiles keep both operands in local memory while a workgroup multiplies them.
const MATMUL_TILE = 16

@kernel inbounds=true function matmul_gpu!(output, left, right, α::A, chunk) where {A}
    row, col, part = @index(Global, NTuple)
    i, j, _ = @index(Local, NTuple)
    a = @localmem A (MATMUL_TILE, MATMUL_TILE)
    b = @localmem A (MATMUL_TILE, MATMUL_TILE)
    total = zero(A)
    first = (part - 1) * chunk
    last = min(first + chunk, size(left, 2))
    for start in first:MATMUL_TILE:last-1
        k, l = start + j, start + i
        a[i, j] = row <= size(left, 1) && k <= last ? A(left[row, k]) : zero(A)
        b[i, j] = col <= size(right, 2) && l <= last ? A(right[l, col]) : zero(A)
        @synchronize
        for inner in 1:MATMUL_TILE
            total = muladd(a[i, inner], b[inner, j], total)
        end
        @synchronize
    end
    if row <= size(output, 1) && col <= size(output, 2)
        output[row, col, part] = α * total
    end
end

@kernel inbounds=true function sum_matmul_partials_gpu!(output, partials, α::A) where {A}
    row, col = @index(Global, NTuple)
    total = zero(A)
    for part in axes(partials, 3)
        total += partials[row, col, part]
    end
    output[row, col] = α * total
end

function gpu_multiply!(output, left, right, ::Type{A}, α) where {A}
    padded = map(n -> cld(n, MATMUL_TILE) * MATMUL_TILE, size(output))
    width = size(left, 2)
    tile = (MATMUL_TILE, MATMUL_TILE)
    # Weight gradients have small outputs and long reductions. Give each
    # output tile several independent workgroups to fill more of the device.
    if width >= 2048 && length(output) <= 2^16
        chunk = cld(width, 32 * MATMUL_TILE) * MATMUL_TILE
        parts = cld(width, chunk)
        partials = similar(output, A, (size(output)..., parts))
        launch!(matmul_gpu!, partials, left, right, one(A), chunk;
                ndrange=(padded..., parts), groupsize=(tile..., 1))
        return launch!(sum_matmul_partials_gpu!, output, partials, A(α);
                       ndrange=size(output), groupsize=tile)
    end
    launch!(matmul_gpu!, reshape(output, size(output)..., 1), left, right, A(α), width;
            ndrange=(padded..., 1), groupsize=(tile..., 1))
    return output
end

@kernel inbounds=true function column_sum_gpu!(output, values, divisor::A) where {A}
    row = @index(Global, Linear)
    total = zero(A)
    for col in axes(values, 2)
        total += A(values[row, col])
    end
    output[row] = total / divisor
end

function column_sum!(output, values, divisor::A) where {A}
    rows, columns = size(values)
    columns <= 512 && return launch!(column_sum_gpu!, output, values, divisor)
    backend = device_of(output)
    # A moderate reduction can stay serial per row when there are enough workgroups.
    if columns <= 1024 && 0 < KA.KernelInterface.multiprocessor_count(backend) <= cld(rows, 256) ÷ 2
        return launch!(column_sum_gpu!, output, values, divisor)
    end
    reduce_sum!(A, output, values; backend)
    output ./= divisor
    return output
end

@kernel inbounds=true function softmax_gpu_kernel!(y, x, ::Type{A}) where {A}
    col = @index(Global, Linear)
    peak = A(-Inf)
    for i in axes(x, 1)
        peak = max(peak, A(x[i, col]))
    end
    total = zero(A)
    for i in axes(x, 1)
        total += exp(A(x[i, col]) - peak)
    end
    for i in axes(x, 1)
        y[i, col] = exp(A(x[i, col]) - peak) / total
    end
end

function gpu_softmax!(output, input)
    width = size(input, 1)
    x, y = reshape(input, width, :), reshape(output, width, :)
    launch!(softmax_gpu_kernel!, y, x, accumulation_type(eltype(x)); ndrange=size(x, 2))
    return output
end

@kernel inbounds=true function softmax_backward_gpu_kernel!(∇x, p, ∇p, ::Type{A}) where {A}
    col = @index(Global, Linear)
    projection = zero(A)
    for i in axes(p, 1)
        projection += A(p[i, col]) * A(∇p[i, col])
    end
    for i in axes(p, 1)
        ∇x[i, col] = A(p[i, col]) * (A(∇p[i, col]) - projection)
    end
end

function gpu_softmax_backward!(output, p, upstream)
    ∇x, y, ∇y = map(x -> reshape(x, size(p, 1), :), (output, p, upstream))
    launch!(softmax_backward_gpu_kernel!, ∇x, y, ∇y, accumulation_type(eltype(p)); ndrange=size(y, 2))
    return output
end

@kernel inbounds=true function layer_norm_gpu!(y, x, z, inverse_std, γ, β, ε::A) where {A}
    col = @index(Global, Linear)
    width = A(size(x, 1))
    total = zero(A)
    for i in axes(x, 1)
        total += A(x[i, col])
    end
    μ = total / width
    squared = zero(A)
    for i in axes(x, 1)
        centered = A(x[i, col]) - μ
        z[i, col] = centered
        squared += abs2(centered)
    end
    σ⁻¹ = inv(sqrt(squared / width + ε))
    inverse_std[col] = σ⁻¹
    for i in axes(x, 1)
        normalized = z[i, col] * σ⁻¹
        z[i, col] = normalized
        y[i, col] = normalized * A(γ[i]) + A(β[i])
    end
end

function gpu_norm_forward!(layer, cache, input)
    width = size(input, 1)
    x, y, z = map(a -> reshape(a, width, :), (input, cache.output, cache.normalized))
    launch!(layer_norm_gpu!, y, x, z, cache.inverse_std, layer.scale, layer.bias, layer.epsilon;
            ndrange=size(x, 2))
    return cache.output
end

@kernel inbounds=true function layer_norm_input_gpu!(∇x, ∇y, z, inverse_std, γ)
    col = @index(Global, Linear)
    A = eltype(z)
    width = A(size(z, 1))
    total, projection = zero(A), zero(A)
    for i in axes(z, 1)
        h = A(∇y[i, col]) * A(γ[i])
        total += h
        projection += h * z[i, col]
    end
    μ, mean_projection = total / width, projection / width
    σ⁻¹ = inverse_std[col]
    for i in axes(z, 1)
        h = A(∇y[i, col]) * A(γ[i])
        ∇x[i, col] = σ⁻¹ * (h - μ - z[i, col] * mean_projection)
    end
end

@kernel inbounds=true function layer_norm_parameters_gpu!(∇γ, ∇β, ∇y, z, batch::A) where {A}
    i = @index(Global, Linear)
    scale, bias = zero(A), zero(A)
    for col in axes(z, 2)
        g = A(∇y[i, col])
        scale += g * z[i, col]
        bias += g
    end
    ∇γ[i] = scale / batch
    ∇β[i] = bias / batch
end

function gpu_norm_backward!(layer, cache, input, upstream, compute_parameters)
    width = size(input, 1)
    ∇x, ∇y, z = map(a -> reshape(a, width, :), (cache.input_gradient, upstream, cache.normalized))
    launch!(layer_norm_input_gpu!, ∇x, ∇y, z, cache.inverse_std, layer.scale; ndrange=size(z, 2))
    if compute_parameters
        batch = eltype(z)(size(input, ndims(input)))
        launch!(layer_norm_parameters_gpu!, cache.scale_gradient, cache.bias_gradient, ∇y, z, batch)
    end
    return cache.input_gradient
end

# Generate dropout samples on-device from the seed owned by the training state.
@inline function random_bits(seed, index)
    z = seed + UInt64(index) * 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

uniform_sample(::Type{Float64}, bits) = Float64(bits >> 11) * 0x1.0p-53
uniform_sample(::Type{Float32}, bits) = Float32(bits >> 40) * Float32(0x1.0p-24)

@kernel inbounds=true function random_uniform_gpu!(output, seed)
    i = @index(Global, Linear)
    output[i] = uniform_sample(eltype(output), random_bits(seed, i))
end
