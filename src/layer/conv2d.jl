using .SpatialConvolution: Workspace, convolve!, conv_im2col!, conv_direct!,
    ∇conv_data_im2col!, ∇conv_data_direct!, ∇conv_filter_im2col!, ∇conv_filter_direct!

struct Convolution{Transposed,T<:AbstractFloat,F<:AbstractActivation} <: AbstractLayer
    filters::Array{T,4}
    bias::Vector{T}
    padding::Int
    stride::NTuple{2,Int}
    activation::F
end

const Conv2D = Convolution{false}
const TConv2D = Convolution{true}

function Convolution{Transposed}(input::Int, output::Int; kernel_size::NTuple{2,Int},
                                  stride::NTuple{2,Int}=(1, 1), padding::Int=0,
                                  activation::AbstractActivation=Identity(),
                                  randomization::Bool=true,
                                  precision::Type{T}=Float32) where {Transposed,T<:AbstractFloat}
    checked_precision(T)
    input > 0 && output > 0 && padding >= 0 && all(>(0), stride) && all(>(0), kernel_size) ||
        throw(ArgumentError("invalid convolution channels, kernel, padding or stride"))
    dims = (output, input, kernel_size...)
    area = prod(kernel_size)
    filters = randomization ? GUMatrix(area * input, area * output, dims, T) : zeros(T, dims)
    return Convolution{Transposed,T,typeof(activation)}(filters, zeros(T, output), padding, stride, activation)
end

precision_type(::Convolution{P,T}) where {P,T} = T
input_shape(layer::Convolution) = (nothing, nothing, size(layer.filters, 2))
layer_parameters(layer::Convolution) = (; filters=layer.filters, bias=layer.bias)

function spatial_shape(input)
    input === nothing && return (nothing, nothing, nothing)
    if !isempty(input) && last(input) isa Colon
        length(input) <= 4 || throw(DimensionMismatch("too many axes for spatial data"))
        return ntuple(i -> i < length(input) ? input[i] : nothing, 3)
    end
    length(input) == 3 || throw(DimensionMismatch("expected width × height × channels before the batch axis"))
    return input
end

convolved_extent(n, k, padding, stride, ::Val{false}) = fld(n + 2 * padding - k, stride) + 1
convolved_extent(n, k, padding, stride, ::Val{true}) = (n - 1) * stride + 2 * padding - k + 2

function output_shape(layer::Convolution{P}, input) where {P}
    dims = spatial_shape(input)
    output = ntuple(2) do d
        dims[d] === nothing && return nothing
        extent = convolved_extent(dims[d], size(layer.filters, d + 2), layer.padding, layer.stride[d], Val(P))
        extent > 0 || throw(DimensionMismatch("convolution has an empty spatial output"))
        return extent
    end
    return (output..., length(layer.bias))
end

mutable struct ConvolutionCache{T,A}
    filters::Array{A,4}
    bias::Vector{A}
    values::Array{T,4}
    output::Array{T,4}
    delta::Array{T,4}
    input_gradient::Array{T,4}
    padded::Array{T,4}
    padded_gradient::Array{T,4}
    workspace::Workspace
    ConvolutionCache{T,A}(filters, bias) where {T,A} = new{T,A}(filters, bias)
end

function layer_cache(layer::Convolution{P,T}) where {P,T}
    A = accumulation_type(T)
    return ConvolutionCache{T,A}(similar(layer.filters, A), similar(layer.bias, A))
end
layer_gradients(::Convolution, cache::ConvolutionCache) = (; filters=cache.filters, bias=cache.bias)

function resize_convolution!(layer::Convolution{P,T}, cache, input) where {P,T}
    isdefined(cache, :input_gradient) && size(cache.input_gradient) == size(input) && return
    dims = (shape(layer, size(input)[1:3])..., size(input, 4))
    cache.values = Array{T}(undef, dims)
    cache.output, cache.delta = similar(cache.values), similar(cache.values)
    cache.input_gradient = similar(input)
    if P
        expanded = ntuple(d -> (size(input, d) - 1) * layer.stride[d] + 2 * layer.padding + 1, 2)
        cache.padded = zeros(T, expanded..., size(input, 3), size(input, 4))
        cache.padded_gradient = similar(cache.padded)
        cache.workspace = Workspace(size(cache.padded), layer.filters; stride=(1, 1), padding=0)
    else
        cache.workspace = Workspace(size(input), layer.filters; stride=layer.stride, padding=layer.padding)
    end
    return
end

strided_indices(layer, input, d) = range(layer.padding + 1; step=layer.stride[d], length=size(input, d))

function forward!(layer::Convolution{P,T}, cache::ConvolutionCache{T}, input::Array{T,4}, mode) where {P,T}
    resize_convolution!(layer, cache, input)
    work = cache.workspace
    source = input
    if P
        source = cache.padded
        @views source[strided_indices(layer, input, 1), strided_indices(layer, input, 2), :, :] .= input
    end
    permutedims!(work.kernel, layer.filters, (3, 4, 2, 1))
    convolve!(conv_im2col!, conv_direct!, cache.values, source, work.kernel, work)
    cache.values .+= reshape(layer.bias, 1, 1, :, 1)
    return activate_values!(cache.output, layer.activation, cache.values)
end

function backward!(layer::Convolution{P,T}, cache::ConvolutionCache{T,A}, input, delta;
                   compute_parameters=true) where {P,T,A}
    size(delta) == size(cache.output) || throw(DimensionMismatch("convolution cotangent shape differs"))
    activation_gradient!(cache.delta, layer.activation, cache.values, cache.output, delta)
    work = cache.workspace
    source = P ? cache.padded : input
    dx = P ? cache.padded_gradient : cache.input_gradient
    convolve!(∇conv_data_im2col!, ∇conv_data_direct!, dx, cache.delta, work.kernel, work)
    if P
        @views cache.input_gradient .= dx[strided_indices(layer, input, 1), strided_indices(layer, input, 2), :, :]
    end
    if compute_parameters
        batch = A(size(delta, 4))
        convolve!(∇conv_filter_im2col!, ∇conv_filter_direct!, work.gradient, source, cache.delta, work; alpha=inv(batch))
        permutedims!(cache.filters, work.gradient, (4, 3, 1, 2))
        for f in axes(delta, 3)
            total = zero(A)
            @inbounds for b in axes(delta, 4), j in axes(delta, 2)
                @simd for i in axes(delta, 1)
                    total += convert(A, cache.delta[i, j, f, b])
                end
            end
            cache.bias[f] = total / batch
        end
    end
    return cache.input_gradient
end
