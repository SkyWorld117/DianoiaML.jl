using .Threads

# DataSet (Batch_Size, ?, ?...)
function flatten(DataSet::Any, batch_dim::Int64)
    dims = ndims(DataSet)
    len = 1
    for i in 1:dims
        if i!=batch_dim
            len *= size(DataSet, i)
        end
    end
    input_data = zeros(Float32, (len, size(DataSet, batch_dim)))
    @threads for i in axes(DataSet, batch_dim)
        if dims==3
            input_data[:,i] = Array{Float32}(reshape(transpose(selectdim(DataSet, batch_dim, i)), (len,1)))
        else
            input_data[:,i] = Array{Float32}(reshape(selectdim(DataSet, batch_dim, i), (len,1)))
        end
    end
    return input_data
end
