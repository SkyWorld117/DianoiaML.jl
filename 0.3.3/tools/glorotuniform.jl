module GlorotUniform
    function GUMatrix(input_size::Int64, layer_size::Int64, shape::Any)
        return rand(-sqrt(6/(input_size+layer_size)):0.00001:sqrt(6/(input_size+layer_size)), shape)
    end
    export GUMatrix
end

# Source: https://keras.io/api/layers/initializers/#glorotuniform-class
