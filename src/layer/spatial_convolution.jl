module SpatialConvolution
using ..DianoiaML: accumulation_type, float_input
using NNlib: DenseConvDims, conv_im2col!, ∇conv_data_im2col!, ∇conv_filter_im2col!,
             conv_direct!, ∇conv_data_direct!, ∇conv_filter_direct!,
             conv_im2col_dims, ∇conv_data_im2col_dims, ∇conv_filter_im2col_dims,
             insert_singleton_spatial_dimension

struct Workspace{D,A,C}
    dims::D
    kernel::Array{A,4}
    gradient::Array{A,4}
    columns::C
end

function Workspace(input_size, filters::Array{T,4}; stride, padding) where {T}
    A = accumulation_type(T)
    kernel = similar(filters, A, size(filters, 3), size(filters, 4), size(filters, 2), size(filters, 1))
    dims = DenseConvDims(input_size, size(kernel); stride, padding, flipkernel=true)
    columns = if A <: Union{Float32,Float64}
        dims3 = insert_singleton_spatial_dimension(dims)
        batch, tasks = input_size[end], Threads.nthreads()
        forward = conv_im2col_dims(dims3, A, batch, tasks)
        data = ∇conv_data_im2col_dims(dims3, A, batch, tasks)
        filter = ∇conv_filter_im2col_dims(dims3, A, batch, tasks)
        Array{A}(undef, max.(forward, data, filter))
    else
        nothing
    end
    return Workspace(dims, kernel, similar(kernel), columns)
end

# The direct kernels work for floating-point types without a BLAS implementation.
function convolve!(fast, generic, dest, left, right, work::Workspace{D,A}; alpha=one(A)) where {D,A}
    result = eltype(dest) === A ? dest : similar(dest, A)
    x, y = float_input(A, left), float_input(A, right)
    if work.columns === nothing
        generic(result, x, y, work.dims; alpha)
    else
        fast(result, x, y, work.dims; col=work.columns, alpha)
    end
    dest === result || copyto!(dest, result)
    return dest
end

end
