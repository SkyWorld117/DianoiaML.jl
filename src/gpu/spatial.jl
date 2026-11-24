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

@kernel inbounds=true function image_to_columns_gpu!(patches, input, output_width, output_height,
                                                      kernel_width, stride, padding)
    feature, position = @index(Global, NTuple)
    channels = size(input, 3)
    k, c = divrem(feature - 1, channels)
    ky, kx = divrem(k, kernel_width)
    i, j, b = spatial_position(position, output_width, output_height)
    x = (i-1) * stride[1] - padding + kx + 1
    y = (j-1) * stride[2] - padding + ky + 1
    valid = 1 <= x <= size(input, 1) && 1 <= y <= size(input, 2)
    patches[feature, position] = valid ? input[x, y, c+1, b] : zero(eltype(patches))
end

@kernel inbounds=true function convolution_output_gpu!(output, values, bias)
    i, j, c, b = @index(Global, NTuple)
    position = i + size(output, 1) * ((j-1) + size(output, 2) * (b-1))
    output[i, j, c, b] = values[c, position] + bias[c]
end

@kernel inbounds=true function convolution_delta_gpu!(values, delta)
    c, position = @index(Global, NTuple)
    i, j, b = spatial_position(position, size(delta, 1), size(delta, 2))
    values[c, position] = delta[i, j, c, b]
end

@kernel inbounds=true function columns_to_image_gpu!(output, patches, output_width, output_height,
                                                      kernel_size, stride, padding)
    x, y, c, b = @index(Global, NTuple)
    channels = size(output, 3)
    total = zero(eltype(patches))
    for ky in 0:kernel_size[2]-1, kx in 0:kernel_size[1]-1
        i, dx = divrem(x + padding - kx - 1, stride[1])
        j, dy = divrem(y + padding - ky - 1, stride[2])
        if iszero(dx) && iszero(dy) && 0 <= i < output_width && 0 <= j < output_height
            feature = c + channels * (kx + kernel_size[1] * ky)
            position = i + 1 + output_width * (j + output_height * (b-1))
            total += patches[feature, position]
        end
    end
    output[x, y, c, b] = total
end

const SPATIAL_GROUP = (16, 8, 1, 1)
const COLUMN_GROUP = (16, 16)

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
    launch!(image_to_columns_gpu!, work.patches, source, width, height, size(layer.filters, 3),
            stride, padding; ndrange=size(work.patches), groupsize=COLUMN_GROUP)
    W = reshape(layer.filters, size(layer.filters, 1), :)
    multiply!(work.values, W, work.patches, eltype(work.values))
    launch!(convolution_output_gpu!, cache.values, work.values, layer.bias;
            ndrange=size(cache.values), groupsize=SPATIAL_GROUP)
    return activate_values!(cache.output, layer.activation, cache.values)
end

function gpu_convolution_backward!(layer::Convolution{P}, cache, input, compute_parameters) where {P}
    work = cache.workspace
    A = eltype(work.values)
    stride, padding = P ? ((1, 1), 0) : (layer.stride, layer.padding)
    launch!(convolution_delta_gpu!, work.values, cache.delta; ndrange=size(work.values), groupsize=COLUMN_GROUP)
    W = reshape(layer.filters, size(layer.filters, 1), :)
    multiply!(work.patch_gradient, transpose(W), work.values, A)
    ∇x = P ? cache.padded_gradient : cache.input_gradient
    kernel_size = (size(layer.filters, 3), size(layer.filters, 4))
    launch!(columns_to_image_gpu!, ∇x, work.patch_gradient, size(cache.delta, 1), size(cache.delta, 2),
            kernel_size, stride, padding; ndrange=size(∇x), groupsize=SPATIAL_GROUP)
    if P
        launch!(extract_spatial_gpu!, cache.input_gradient, ∇x, layer.stride, layer.padding;
                ndrange=size(input), groupsize=SPATIAL_GROUP)
    end
    if compute_parameters
        batch = A(size(input, 4))
        multiply!(reshape(cache.filters, size(W)), work.values, transpose(work.patches), A, inv(batch))
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
