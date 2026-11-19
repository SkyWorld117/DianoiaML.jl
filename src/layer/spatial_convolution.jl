module SpatialConvolution
using NNlib: DenseConvDims, conv_im2col!, ∇conv_data_im2col!, ∇conv_filter_im2col!,
             conv_im2col_dims, ∇conv_data_im2col_dims, ∇conv_filter_im2col_dims,
             insert_singleton_spatial_dimension
using ..DianoiaML: update_parameters!

struct Workspace{D}
    dims::D
    kernel::Array{Float32,4}
    gradient::Array{Float32,4}
    columns::Array{Float32,3}
end

function Workspace(input_size, filters; stride, padding)
    kernel = similar(filters, size(filters, 3), size(filters, 4), size(filters, 2), size(filters, 1))
    dims = DenseConvDims(input_size, size(kernel); stride, padding, flipkernel=true)
    dims3 = insert_singleton_spatial_dimension(dims)
    batch, tasks = input_size[end], Threads.nthreads()
    forward = conv_im2col_dims(dims3, Float32, batch, tasks)
    data = ∇conv_data_im2col_dims(dims3, Float32, batch, tasks)
    filter = ∇conv_filter_im2col_dims(dims3, Float32, batch, tasks)
    # These layers are ungrouped, so call the backend directly and reuse one
    # patch matrix across the forward pass and both gradient calculations.
    columns = Array{Float32}(undef, max.(forward, data, filter))
    return Workspace(dims, kernel, similar(kernel), columns)
end

function forward!(layer, input, work::Workspace=layer.workspace)
    permutedims!(work.kernel, layer.filters, (3, 4, 2, 1))
    conv_im2col!(layer.value, input, work.kernel, work.dims; col=work.columns)
    layer.value .+= reshape(layer.biases, 1, 1, :, 1)
    layer.activation_function.func!(layer.output, layer.value)
end

function backward!(layer, input, input_gradient, optimizer, α, parameters, direction, work::Workspace=layer.workspace)
    ∇conv_data_im2col!(input_gradient, layer.∇biases, work.kernel, work.dims; col=work.columns)
    direction == 0 && return

    delta = layer.∇biases
    batch = size(delta, 4)
    ∇conv_filter_im2col!(work.gradient, input, delta, work.dims;
                        col=work.columns, alpha=inv(Float32(batch)))
    permutedims!(layer.∇weights, work.gradient, (4, 3, 1, 2))
    for f in axes(delta, 3)
        total = 0f0
        @inbounds for b in axes(delta, 4), j in axes(delta, 2)
            @simd for i in axes(delta, 1)
                total += delta[i, j, f, b]
            end
        end
        layer.∇biases_c[f] = total / batch
    end

    rate = Float32(α * direction)
    update_parameters!(optimizer, layer.filters, layer.∇weights, layer.Vdw, layer.Sdw, rate, parameters)
    update_parameters!(optimizer, layer.biases, layer.∇biases_c, layer.Vdb, layer.Sdb, rate, parameters)
end
end
