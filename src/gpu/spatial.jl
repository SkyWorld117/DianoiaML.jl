function device_convolution_workspace(cache)
    channels = size(cache.values, 3)
    positions = length(cache.values) ÷ channels
    dims = (length(cache.filters) ÷ channels, positions)
    patches = similar(cache.filters, eltype(cache.filters), dims)
    return DeviceConvolutionWorkspace(patches, similar(patches),
                                     similar(patches, eltype(patches), (channels, positions)))
end

@kernel inbounds=true function expand_spatial_gpu!(output, input, stride, padding)
    i, j, c, b = @index(Global, NTuple)
    x, dx = divrem(i - padding - 1, stride[1])
    y, dy = divrem(j - padding - 1, stride[2])
    valid = iszero(dx) && iszero(dy) && 0 <= x < size(input, 1) && 0 <= y < size(input, 2)
    output[i, j, c, b] = valid ? input[x+1, y+1, c, b] : zero(eltype(output))
end

@kernel inbounds=true function extract_spatial_gpu!(output, input, stride, padding)
    i, j, c, b = @index(Global, NTuple)
    output[i, j, c, b] = input[(i-1) * stride[1] + padding + 1, (j-1) * stride[2] + padding + 1, c, b]
end

@inline function spatial_position(position, width, height)
    n, x = divrem(position - 1, width)
    b, y = divrem(n, height)
    return x + 1, y + 1, b + 1
end

@inline function image_patch_value(input, feature, position, output_width, output_height, kernel_width, stride, padding)
    channels = size(input, 3)
    k, c = divrem(feature - 1, channels)
    ky, kx = divrem(k, kernel_width)
    i, j, b = spatial_position(position, output_width, output_height)
    x = (i-1) * stride[1] - padding + kx + 1
    y = (j-1) * stride[2] - padding + ky + 1
    valid = 1 <= x <= size(input, 1) && 1 <= y <= size(input, 2)
    return valid ? (@inbounds input[x, y, c+1, b]) : zero(eltype(input))
end

@kernel inbounds=true function image_to_columns_gpu!(patches, input, width, height, kernel_width, stride, padding)
    feature, position = @index(Global, NTuple)
    patches[feature, position] = image_patch_value(input, feature, position, width, height, kernel_width, stride, padding)
end

const PATCH_TILE = 16

# Load adjacent image pixels and store adjacent patch features. Local memory
# changes which dimension consecutive lanes access between the two passes.
@kernel inbounds=true function image_to_columns_tiled_gpu!(patches, input, width, height, kernel_width,
                                                           stride, padding, ::Type{A}) where {A}
    feature, position = @index(Global, NTuple)
    i, j = @index(Local, NTuple)
    tile = @localmem A (PATCH_TILE + 1, PATCH_TILE)
    f, p = feature - i + j, position - j + i
    if f <= size(patches, 1) && p <= size(patches, 2)
        tile[j, i] = image_patch_value(input, f, p, width, height, kernel_width, stride, padding)
    end
    @synchronize
    if feature <= size(patches, 1) && position <= size(patches, 2)
        patches[feature, position] = tile[i, j]
    end
end

@kernel inbounds=true function convolution_output_gpu!(output, z, values, bias, activation, args)
    i, j, c, b = @index(Global, NTuple)
    position = i + size(output, 1) * ((j-1) + size(output, 2) * (b-1))
    value = eltype(z)(values[c, position] + bias[c])
    z[i, j, c, b] = value
    output[i, j, c, b] = activate_scalar(activation, value, args)
end

@kernel inbounds=true function convolution_delta_gpu!(values, delta)
    c, position = @index(Global, NTuple)
    i, j, b = spatial_position(position, size(delta, 1), size(delta, 2))
    values[c, position] = delta[i, j, c, b]
end

@kernel inbounds=true function convolution_activation_delta_gpu!(values, upstream, z, y, activation, args)
    c, position = @index(Global, NTuple)
    x, j, b = spatial_position(position, size(z, 1), size(z, 2))
    ∇y = upstream[x, j, c, b]
    ∇z = pointwise_pullback(activation, z[x, j, c, b], y[x, j, c, b], ∇y, args)
    values[c, position] = eltype(z)(∇z)
end

@inline function patch_gradient_value(patches, x, y, c, b, channels, output_width, output_height,
                                      kernel_size, ::Val{stride}, padding) where {stride}
    total = zero(eltype(patches))
    for ky in 0:kernel_size[2]-1
        j, dy = divrem(y + padding - ky - 1, stride[2])
        if iszero(dy) && 0 <= j < output_height
            feature = c + channels * kernel_size[1] * ky
            position = 1 + output_width * (j + output_height * (b-1))
            for kx in 0:kernel_size[1]-1
                i, dx = divrem(x + padding - kx - 1, stride[1])
                if iszero(dx) && 0 <= i < output_width
                    @inbounds total += patches[feature + channels * kx, position + i]
                end
            end
        end
    end
    return total
end

@kernel inbounds=true function columns_to_image_gpu!(output, patches, width, height, kernel_size, stride, padding)
    x, y, c, b = @index(Global, NTuple)
    output[x, y, c, b] = patch_gradient_value(patches, x, y, c, b, size(output, 3), width, height,
                                             kernel_size, stride, padding)
end

@kernel inbounds=true function columns_to_image_tiled_gpu!(output, patches, width, height, kernel_size,
                                                           stride, padding, positions, ::Type{T}) where {T}
    channel, position = @index(Global, NTuple)
    i, j = @index(Local, NTuple)
    tile = @localmem T (PATCH_TILE + 1, PATCH_TILE)
    channels = size(output, 3)
    if channel <= channels && position <= positions
        x, y, b = spatial_position(position, size(output, 1), size(output, 2))
        tile[i, j] = patch_gradient_value(patches, x, y, channel, b, channels, width, height,
                                          kernel_size, stride, padding)
    end
    @synchronize
    c, p = channel - i + j, position - j + i
    if c <= channels && p <= positions
        x, y, b = spatial_position(p, size(output, 1), size(output, 2))
        output[x, y, c, b] = tile[j, i]
    end
end

const SPATIAL_GROUP = (16, 8, 1, 1)
const COLUMN_GROUP = (PATCH_TILE, PATCH_TILE)

padded_patch_dims(dims) = map(dim -> cld(dim, PATCH_TILE) * PATCH_TILE, dims)

function pack_image!(patches, input, width, height, kernel_width, stride, padding)
    if size(input, 3) >= PATCH_TILE && size(patches, 2) >= 1024
        return launch!(image_to_columns_tiled_gpu!, patches, input, width, height, kernel_width,
                       stride, padding, eltype(patches);
                       ndrange=padded_patch_dims(size(patches)), groupsize=COLUMN_GROUP)
    end
    return launch!(image_to_columns_gpu!, patches, input, width, height, kernel_width, stride, padding;
                   ndrange=size(patches), groupsize=COLUMN_GROUP)
end

function unpack_patch_gradients!(output, patches, width, height, kernel_size, stride, padding)
    channels = size(output, 3)
    positions = length(output) ÷ channels
    if channels >= PATCH_TILE && positions >= 1024
        return launch!(columns_to_image_tiled_gpu!, output, patches, width, height, kernel_size,
                       stride, padding, positions, eltype(output);
                       ndrange=padded_patch_dims((channels, positions)), groupsize=COLUMN_GROUP)
    end
    return launch!(columns_to_image_gpu!, output, patches, width, height, kernel_size, stride, padding;
                   ndrange=size(output), groupsize=SPATIAL_GROUP)
end

gpu_convolution_multiply!(backend, output, left, right, α::A) where {A} =
    gpu_multiply!(output, left, right, A, α)

function gpu_convolution_forward!(layer::Convolution{P}, cache, input) where {P}
    work = cache.workspace
    stride, padding = P ? ((1, 1), 0) : (layer.stride, layer.padding)
    source = if P
        launch!(expand_spatial_gpu!, cache.padded, input, layer.stride, layer.padding;
                ndrange=size(cache.padded), groupsize=SPATIAL_GROUP)
    else
        input
    end
    width, height = size(cache.values, 1), size(cache.values, 2)
    pack_image!(work.patches, source, width, height, size(layer.filters, 3), stride, padding)
    W = reshape(layer.filters, size(layer.filters, 1), :)
    gpu_convolution_multiply!(device_of(input), work.values, W, work.patches, one(eltype(work.values)))
    if layer.activation isa PointwiseActivation
        return launch!(convolution_output_gpu!, cache.output, cache.values, work.values, layer.bias,
                       layer.activation, pointwise_arguments(layer.activation, eltype(cache.values));
                       ndrange=size(cache.values), groupsize=SPATIAL_GROUP)
    end
    launch!(convolution_output_gpu!, cache.values, cache.values, work.values, layer.bias, Identity(), ();
            ndrange=size(cache.values), groupsize=SPATIAL_GROUP)
    return activate_values!(cache.output, layer.activation, cache.values)
end

function gpu_convolution_backward!(layer::Convolution{P}, cache, input, upstream, compute_parameters) where {P}
    work = cache.workspace
    A = eltype(work.values)
    backend = device_of(input)
    stride, padding = P ? ((1, 1), 0) : (layer.stride, layer.padding)
    if layer.activation isa PointwiseActivation
        launch!(convolution_activation_delta_gpu!, work.values, upstream, cache.values, cache.output,
                layer.activation, pointwise_arguments(layer.activation, eltype(cache.values));
                ndrange=size(work.values), groupsize=COLUMN_GROUP)
    else
        activation_gradient!(cache.delta, layer.activation, cache.values, cache.output, upstream)
        launch!(convolution_delta_gpu!, work.values, cache.delta; ndrange=size(work.values), groupsize=COLUMN_GROUP)
    end
    W = reshape(layer.filters, size(layer.filters, 1), :)
    gpu_convolution_multiply!(backend, work.patch_gradient, transpose(W), work.values, one(A))
    ∇x = P ? cache.padded_gradient : cache.input_gradient
    kernel_size = (size(layer.filters, 3), size(layer.filters, 4))
    unpack_patch_gradients!(∇x, work.patch_gradient, size(cache.values, 1), size(cache.values, 2),
                            kernel_size, Val(stride), padding)
    if P
        launch!(extract_spatial_gpu!, cache.input_gradient, ∇x, layer.stride, layer.padding;
                ndrange=size(input), groupsize=SPATIAL_GROUP)
    end
    if compute_parameters
        batch = A(size(input, 4))
        gpu_convolution_multiply!(backend, reshape(cache.filters, size(W)), work.values,
                                  transpose(work.patches), inv(batch))
        column_sum!(cache.bias, work.values, batch)
    end
    return cache.input_gradient
end

@kernel inbounds=true function max_pool_gpu!(output, indices, input, window, stride)
    i, j, c, b = @index(Global, NTuple)
    x, y = (i-1) * stride[1] + 1, (j-1) * stride[2] + 1
    best = input[x, y, c, b]
    best_x, best_y = x, y
    for dy in 0:window[2]-1, dx in 0:window[1]-1
        pixel = input[x+dx, y+dy, c, b]
        if isless(best, pixel)
            best, best_x, best_y = pixel, x+dx, y+dy
        end
    end
    output[i, j, c, b] = best
    indices[i, j, c, b] = best_x + size(input, 1) * (best_y-1 + size(input, 2) * (c-1 + size(input, 3) * (b-1)))
end

gpu_pool_forward!(layer, cache, input) =
    launch!(max_pool_gpu!, cache.output, cache.indices, input, layer.window, layer.stride;
            ndrange=size(cache.output), groupsize=SPATIAL_GROUP)

@kernel inbounds=true function pool_backward_gpu!(output, upstream, indices, window, stride)
    x, y, c, b = @index(Global, NTuple)
    idx = x + size(output, 1) * (y-1 + size(output, 2) * (c-1 + size(output, 3) * (b-1)))
    A = accumulation_type(eltype(upstream))
    total = zero(A)
    for j in max(1, cld(y-window[2], stride[2])+1):min(size(upstream, 2), fld(y-1, stride[2])+1),
        i in max(1, cld(x-window[1], stride[1])+1):min(size(upstream, 1), fld(x-1, stride[1])+1)
        if indices[i, j, c, b] == idx
            total += A(upstream[i, j, c, b])
        end
    end
    output[x, y, c, b] = total
end

gpu_pool_backward!(layer, cache, upstream) =
    launch!(pool_backward_gpu!, cache.input_gradient, upstream, cache.indices, layer.window, layer.stride;
            ndrange=size(cache.input_gradient), groupsize=SPATIAL_GROUP)

@kernel inbounds=true function upsample_gpu!(output, input, scale)
    x, y, c, b = @index(Global, NTuple)
    output[x, y, c, b] = input[cld(x, scale[1]), cld(y, scale[2]), c, b]
end

@kernel inbounds=true function upsample_backward_gpu!(output, delta, scale)
    x, y, c, b = @index(Global, NTuple)
    A = accumulation_type(eltype(delta))
    total = zero(A)
    for dy in 1:scale[2], dx in 1:scale[1]
        total += A(delta[(x-1) * scale[1] + dx, (y-1) * scale[2] + dy, c, b])
    end
    output[x, y, c, b] = total
end
