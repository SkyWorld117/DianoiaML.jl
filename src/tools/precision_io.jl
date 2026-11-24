using HDF5: Group, create_group, h5open

function stored_precision(name)
    for T in (Float16, BFloat16, Float32, Float64, BigFloat)
        name == string(T) && return T
    end
    throw(ArgumentError("unsupported stored precision $name"))
end

write_array(file, key, values) = write(file, key, values)
write_number(file, key, value::Union{Float16,BFloat16,BigFloat}) = write_array(file, key, [value])
write_number(file, key, value) = write(file, key, value)

function read_number(file, key)
    value = read_array(file, key)
    return value isa AbstractArray ? only(value) : value
end

function write_array(file, key, values::Array{T}) where {T<:Union{Float16,BFloat16,BigFloat}}
    group = create_group(file, key)
    try
        write(group, "dtype", string(T))
        write(group, "shape", collect(size(values)))
        if T === BigFloat
            write(group, "bits", precision.(vec(values)))
            write(group, "values", string.(vec(values)))
        else
            write(group, "bits", collect(reinterpret(UInt16, vec(values))))
        end
    finally
        close(group)
    end
end

function read_array(file, key)
    node = file[key]
    try
        node isa Group || return read(node)
        T = stored_precision(read(node, "dtype"))
        shape = Tuple(Int.(read(node, "shape")))
        bits = read(node, "bits")
        if T === BigFloat
            values = read(node, "values")
            length(values) == length(bits) == prod(shape) ||
                throw(DimensionMismatch("invalid BigFloat array at $key"))
            return reshape([BigFloat(value, RoundNearest; precision=p) for (value, p) in zip(values, bits)], shape)
        end
        T in (Float16, BFloat16) || throw(ArgumentError("unsupported array encoding at $key"))
        return reshape(collect(reinterpret(T, bits)), shape)
    finally
        close(node)
    end
end
