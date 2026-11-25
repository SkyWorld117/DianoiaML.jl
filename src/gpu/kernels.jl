@kernel inbounds=true function broadcast_tensor_gpu!(dest, bc)
    i = @index(Global, Linear)
    index = CartesianIndices(dest)[i]
    dest[index] = bc[index]
end

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

const MATMUL_TILE = 16

@inline function load_matmul_tile!(a, b, left, right, i, j, row, col, start, last, ::Val{B}) where {B}
    A = eltype(a)
    @inbounds for offset in 0:B-1
        step = offset * MATMUL_TILE
        if left isa Transpose
            r, k = row + j + step, start + i
            a[j + step, i] = r <= size(left, 1) && k <= last ? A(left[r, k]) : zero(A)
        else
            r, k = row + i + step, start + j
            a[i + step, j] = r <= size(left, 1) && k <= last ? A(left[r, k]) : zero(A)
        end
        if right isa Transpose
            k, c = start + j, col + i + step
            b[j, i + step] = k <= last && c <= size(right, 2) ? A(right[k, c]) : zero(A)
        else
            k, c = start + i, col + j + step
            b[i, j + step] = k <= last && c <= size(right, 2) ? A(right[k, c]) : zero(A)
        end
    end
    return
end

@inline matmul_result!(output, row, col, part, value, ::Nothing) =
    (@inbounds output[row, col, part] = value)

@inline function matmul_result!(output, row, col, part, value, epilogue::Tuple)
    y, bias, activation, args = epilogue
    T = eltype(output)
    @inbounds z = T(T(value) + bias[row])
    @inbounds output[row, col, part] = z
    @inbounds y[row, col] = activate_scalar(activation, z, args)
    return
end

@inline function store_matmul_result!(output, row, col, part, value, epilogue)
    if row <= size(output, 1) && col <= size(output, 2)
        matmul_result!(output, row, col, part, value, epilogue)
    end
    return
end

# A larger output tile reuses the same shared operands across four accumulators
# per thread. The extra shared-memory row avoids bank conflicts on transposed loads.
@kernel inbounds=true function matmul_gpu!(output, left, right, α::A, chunk, epilogue,
                                           bias_gradient, block::Val{B}) where {A,B}
    block_row, block_col, part = @index(Group, NTuple)
    i, j, _ = @index(Local, NTuple)
    a = @localmem A (B * MATMUL_TILE + 1, MATMUL_TILE)
    b = @localmem A (MATMUL_TILE + 1, B * MATMUL_TILE)
    row = (block_row - 1) * B * MATMUL_TILE
    col = (block_col - 1) * B * MATMUL_TILE
    c11 = c21 = c12 = c22 = zero(A)
    bias1 = bias2 = zero(A)
    reduce_bias = bias_gradient !== nothing && block_col == 1 && j == 1
    first = (part - 1) * chunk
    last = min(first + chunk, size(left, 2))
    for start in first:MATMUL_TILE:last-1
        load_matmul_tile!(a, b, left, right, i, j, row, col, start, last, block)
        @synchronize
        for inner in 1:MATMUL_TILE
            a1, b1 = a[i, inner], b[inner, j]
            c11 = muladd(a1, b1, c11)
            if reduce_bias
                bias1 += a1
            end
            if B == 2
                a2, b2 = a[i + MATMUL_TILE, inner], b[inner, j + MATMUL_TILE]
                c21 = muladd(a2, b1, c21)
                c12 = muladd(a1, b2, c12)
                c22 = muladd(a2, b2, c22)
                if reduce_bias
                    bias2 += a2
                end
            end
        end
        @synchronize
    end
    r, c = row + i, col + j
    store_matmul_result!(output, r, c, part, α * c11, epilogue)
    if B == 2
        store_matmul_result!(output, r + MATMUL_TILE, c, part, α * c21, epilogue)
        store_matmul_result!(output, r, c + MATMUL_TILE, part, α * c12, epilogue)
        store_matmul_result!(output, r + MATMUL_TILE, c + MATMUL_TILE, part, α * c22, epilogue)
    end
    if reduce_bias
        dest, divisor = bias_gradient
        if r <= length(dest)
            dest[r] = bias1 / divisor
        end
        if B == 2 && r + MATMUL_TILE <= length(dest)
            dest[r + MATMUL_TILE] = bias2 / divisor
        end
    end
end

@kernel inbounds=true function sum_matmul_partials_gpu!(output, partials, α::A, epilogue) where {A}
    row, col = @index(Global, NTuple)
    total = zero(A)
    for part in axes(partials, 3)
        total += partials[row, col, part]
    end
    matmul_result!(output, row, col, 1, α * total, epilogue)
end

function gpu_multiply!(output, left, right, ::Type{A}, α; epilogue=nothing, bias_gradient=nothing) where {A}
    width = size(left, 2)
    tile = (MATMUL_TILE, MATMUL_TILE)
    dest = reshape(output, size(output)..., 1)
    # Weight gradients have small outputs and long reductions. Give each
    # output tile several independent workgroups to fill more of the device.
    if bias_gradient === nothing && width >= 2048 && length(output) <= 2^16
        padded = map(n -> cld(n, MATMUL_TILE) * MATMUL_TILE, size(output))
        chunk = cld(width, 32 * MATMUL_TILE) * MATMUL_TILE
        parts = cld(width, chunk)
        partials = similar(output, A, (size(output)..., parts))
        launch!(matmul_gpu!, partials, left, right, one(A), chunk, nothing, nothing, Val(1);
                ndrange=(padded..., parts), groupsize=(tile..., 1))
        launch!(sum_matmul_partials_gpu!, dest, partials, A(α), epilogue;
                ndrange=size(output), groupsize=tile)
        return output
    end
    # Keep smaller matrices distributed across enough workgroups.
    rows, columns = size(output)
    block = width >= 32 && min(rows, columns) >= 32 && length(output) >= 2^18 ? Val(2) : Val(1)
    matmul_launch!(dest, left, right, A(α), width, epilogue, bias_gradient, block, tile)
    return output
end

function matmul_launch!(dest, left, right, α, width, epilogue, bias_gradient, block::Val{B}, tile) where {B}
    padded = map(n -> cld(n, B * MATMUL_TILE) * MATMUL_TILE, (size(dest, 1), size(dest, 2)))
    launch!(matmul_gpu!, dest, left, right, α, width, epilogue, bias_gradient, block;
            ndrange=(padded..., 1), groupsize=(tile..., 1))
    return
end

@kernel inbounds=true function column_sum_gpu!(output, values, divisor::A) where {A}
    row = @index(Global, Linear)
    total = zero(A)
    for col in axes(values, 2)
        total += A(values[row, col])
    end
    output[row] = total / divisor
end

const COLUMN_SUM_GROUP = (8, 16)

@kernel inbounds=true function column_sum_tiled_gpu!(output, values, divisor::A) where {A}
    row, _ = @index(Global, NTuple)
    feature, lane = @index(Local, NTuple)
    scratch = @localmem A COLUMN_SUM_GROUP
    total = zero(A)
    if row <= length(output)
        for col in lane:COLUMN_SUM_GROUP[2]:size(values, 2)
            total += A(values[row, col])
        end
    end
    scratch[feature, lane] = total
    @synchronize
    for offset in (8, 4, 2, 1)
        if lane <= offset
            scratch[feature, lane] += scratch[feature, lane + offset]
        end
        @synchronize
    end
    if lane == 1 && row <= length(output)
        output[row] = scratch[feature, 1] / divisor
    end
end

function column_sum!(output, values, divisor::A) where {A}
    rows, columns = size(values)
    if rows <= 512 && 128 <= columns <= 512
        features, lanes = COLUMN_SUM_GROUP
        padded = cld(rows, features) * features
        return launch!(column_sum_tiled_gpu!, output, values, divisor;
                       ndrange=(padded, lanes), groupsize=COLUMN_SUM_GROUP)
    end
    columns <= 512 && return launch!(column_sum_gpu!, output, values, divisor)
    backend = device_of(output)
    # A moderate reduction can stay serial per row when there are enough workgroups.
    if columns <= 1024 && 0 < KA.KernelInterface.multiprocessor_count(backend) <= cld(rows, 256) ÷ 2
        return launch!(column_sum_gpu!, output, values, divisor)
    end
    reduce_sum!(A, output, values; backend)
    return broadcast_tensor!(/, output, output, divisor)
end

@kernel inbounds=true function batch_norm_statistics_gpu!(inv_std, running_mean, running_variance,
                                                           mean, squared, count, rate, correction, ε)
    c = @index(Global, Linear)
    σ² = squared[c] / count
    inv_std[c] = inv(sqrt(σ² + ε))
    running_mean[c] = (1 - rate) * running_mean[c] + rate * mean[c]
    running_variance[c] = (1 - rate) * running_variance[c] + (rate * correction) * σ²
end

@kernel inbounds=true function batch_norm_affine_gpu!(output, normalized, input, mean, inv_std,
                                                       scale, bias, ::Val{training}) where {training}
    x, y, c, b = @index(Global, NTuple)
    A = eltype(normalized)
    centered = training ? normalized[x, y, c, b] : A(input[x, y, c, b]) - mean[c]
    value = centered * inv_std[c]
    normalized[x, y, c, b] = value
    output[x, y, c, b] = value * scale[c] + bias[c]
end

function gpu_batch_norm_forward!(layer::BatchNorm2D{T,A}, cache, input) where {T,A}
    z, μ, σ², inv_std = cache.normalized, cache.mean, cache.variance, cache.inverse_std
    if cache.training
        count = length(input) ÷ size(input, 3)
        n = A(count)
        reduce_sum!(A, μ, input)
        broadcast_tensor!(/, μ, μ, n)
        broadcast_tensor!((x, mean) -> A(x) - mean, z, input, μ)
        reduce_sum!(abs2, σ², z)
        correction = count > 1 ? n / A(count - 1) : one(A)
        launch!(batch_norm_statistics_gpu!, inv_std, layer.running_mean, layer.running_variance,
                μ, σ², n, layer.momentum, correction, layer.epsilon)
    else
        broadcast_tensor!((variance, ε) -> inv(sqrt(variance + ε)), inv_std,
                          channel_axes(layer.running_variance), layer.epsilon)
    end
    return launch!(batch_norm_affine_gpu!, cache.output, z, input, layer.running_mean, inv_std,
                   layer.scale, layer.bias, Val(cache.training); ndrange=size(input), groupsize=SPATIAL_GROUP)
end

@kernel inbounds=true function batch_norm_parameter_gradients_gpu!(∇γ, ∇β, sum_product, sum_upstream, batch)
    c = @index(Global, Linear)
    ∇γ[c] = sum_product[c] / batch
    ∇β[c] = sum_upstream[c] / batch
end

@kernel inbounds=true function dense_activation_gradient_gpu!(out, z, y, upstream, activation, args)
    i = @index(Global, Linear)
    out[i] = pointwise_pullback(activation, z[i], y[i], upstream[i], args)
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
    columns = size(x, 2)
    if width >= NORM_GROUP
        launch!(layer_norm_rows_gpu!, y, x, z, cache.inverse_std, layer.scale, layer.bias, layer.epsilon;
                ndrange=NORM_GROUP * columns, groupsize=NORM_GROUP)
    else
        launch!(layer_norm_gpu!, y, x, z, cache.inverse_std, layer.scale, layer.bias, layer.epsilon;
                ndrange=columns)
    end
    return cache.output
end

const NORM_GROUP = 64

@kernel inbounds=true function layer_norm_rows_gpu!(y, x, z, inverse_std, γ, β, ε::A) where {A}
    lane = @index(Local, Linear)
    col = @index(Group, Linear)
    scratch = @localmem A (NORM_GROUP,)
    width = size(x, 1)
    total = zero(A)
    for i in lane:NORM_GROUP:width
        total += A(x[i, col])
    end
    scratch[lane] = total
    @synchronize
    for offset in (32, 16, 8, 4, 2, 1)
        if lane <= offset
            scratch[lane] += scratch[lane + offset]
        end
        @synchronize
    end
    μ = scratch[1] / A(width)
    @synchronize
    squared = zero(A)
    for i in lane:NORM_GROUP:width
        squared += abs2(A(x[i, col]) - μ)
    end
    scratch[lane] = squared
    @synchronize
    for offset in (32, 16, 8, 4, 2, 1)
        if lane <= offset
            scratch[lane] += scratch[lane + offset]
        end
        @synchronize
    end
    σ⁻¹ = inv(sqrt(scratch[1] / A(width) + ε))
    if lane == 1
        inverse_std[col] = σ⁻¹
    end
    for i in lane:NORM_GROUP:width
        normalized = (A(x[i, col]) - μ) * σ⁻¹
        z[i, col] = normalized
        y[i, col] = normalized * A(γ[i]) + A(β[i])
    end
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

@kernel inbounds=true function layer_norm_rows_input_gpu!(∇x, ∇y, z, inverse_std, γ, ::Type{A}) where {A}
    lane = @index(Local, Linear)
    col = @index(Group, Linear)
    sums = @localmem A (NORM_GROUP,)
    projections = @localmem A (NORM_GROUP,)
    width = size(z, 1)
    total, projection = zero(A), zero(A)
    for i in lane:NORM_GROUP:width
        h = A(∇y[i, col]) * A(γ[i])
        total += h
        projection += h * z[i, col]
    end
    sums[lane], projections[lane] = total, projection
    @synchronize
    for offset in (32, 16, 8, 4, 2, 1)
        if lane <= offset
            sums[lane] += sums[lane + offset]
            projections[lane] += projections[lane + offset]
        end
        @synchronize
    end
    μ, mean_projection = sums[1] / A(width), projections[1] / A(width)
    σ⁻¹ = inverse_std[col]
    for i in lane:NORM_GROUP:width
        h = A(∇y[i, col]) * A(γ[i])
        ∇x[i, col] = σ⁻¹ * (h - μ - z[i, col] * mean_projection)
    end
end

const NORM_PARAMETER_GROUP = (8, 16)

@kernel inbounds=true function layer_norm_columns_gpu!(∇γ, ∇β, ∇y, z, batch::A) where {A}
    i, _ = @index(Global, NTuple)
    feature, lane = @index(Local, NTuple)
    scales = @localmem A NORM_PARAMETER_GROUP
    biases = @localmem A NORM_PARAMETER_GROUP
    scale, bias = zero(A), zero(A)
    if i <= size(z, 1)
        for col in lane:NORM_PARAMETER_GROUP[2]:size(z, 2)
            g = A(∇y[i, col])
            scale += g * z[i, col]
            bias += g
        end
    end
    scales[feature, lane], biases[feature, lane] = scale, bias
    @synchronize
    for offset in (8, 4, 2, 1)
        if lane <= offset
            scales[feature, lane] += scales[feature, lane + offset]
            biases[feature, lane] += biases[feature, lane + offset]
        end
        @synchronize
    end
    if lane == 1 && i <= size(z, 1)
        ∇γ[i] = scales[feature, 1] / batch
        ∇β[i] = biases[feature, 1] / batch
    end
end

function gpu_norm_backward!(layer, cache, input, upstream, compute_parameters)
    width = size(input, 1)
    ∇x, ∇y, z = map(a -> reshape(a, width, :), (cache.input_gradient, upstream, cache.normalized))
    columns = size(z, 2)
    if width >= NORM_GROUP
        launch!(layer_norm_rows_input_gpu!, ∇x, ∇y, z, cache.inverse_std, layer.scale, eltype(z);
                ndrange=NORM_GROUP * columns, groupsize=NORM_GROUP)
    else
        launch!(layer_norm_input_gpu!, ∇x, ∇y, z, cache.inverse_std, layer.scale; ndrange=columns)
    end
    if compute_parameters
        batch = eltype(z)(size(input, ndims(input)))
        if columns >= 128
            padded = cld(width, NORM_PARAMETER_GROUP[1]) * NORM_PARAMETER_GROUP[1]
            launch!(layer_norm_columns_gpu!, cache.scale_gradient, cache.bias_gradient, ∇y, z, batch;
                    ndrange=(padded, NORM_PARAMETER_GROUP[2]), groupsize=NORM_PARAMETER_GROUP)
        else
            launch!(layer_norm_parameters_gpu!, cache.scale_gradient, cache.bias_gradient, ∇y, z, batch)
        end
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
