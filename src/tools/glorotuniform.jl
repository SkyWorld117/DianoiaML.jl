module GlorotUniform
    function GUMatrix(input_size::Int, layer_size::Int, shape)
        limit = sqrt(6f0 / (input_size + layer_size))
        values = rand(Float32, shape)
        values .= (2f0 * limit) .* values .- limit
        return values
    end
    export GUMatrix
end

# Source: https://keras.io/api/layers/initializers/#glorotuniform-class
