using .Threads

function flatten(Dataset::Any, batch_dim::Int64)
    dims = ndims(Dataset)
    len = 1
    for i in 1:dims
        if i!=batch_dim
            len *= size(Dataset, i)
        end
    end
    input_data = zeros(Float32, (len, size(Dataset, batch_dim)))
    @threads for i in axes(Dataset, batch_dim)
        input_data[:,i] = Array{Float32}(reshape(selectdim(Dataset, batch_dim, i), (len,1)))
    end
    return input_data
end
