using .Threads

# DataSet (Batch_Size, ?, ?...)
function flatten(DataSet::Array)
    dims = ndims(DataSet)
    len = 1
    for i in 2:dims
        len *= size(DataSet, i)
    end
    input_data = zeros(Float32, (len, size(DataSet, 1)))
    Threads.@threads for i in 1:size(DataSet, 1)
        if dims==3
            input_data[:,i] = Array{Float32}(reshape(transpose(selectdim(DataSet, 1, i)), (len,1)))
        else
            input_data[:,i] = Array{Float32}(reshape(selectdim(DataSet, 1, i), (len,1)))
        end
    end
    return input_data
end
