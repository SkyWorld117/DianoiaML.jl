"""
    TransformerBlock(; input_shape, heads, hidden_size=4*first(input_shape),
                     causal=false, padding_mask=false, dropout=0,
                     epsilon=1//100000, precision=Float32)

Build a pre-norm Transformer block from attention and GELU feed-forward
residual branches. For padded batches, use `padding_mask=true` and supply
`(features, key_mask)` to the resulting graph. The Boolean key mask has shape
`(tokens, examples)`. Positive dropout applies to both residual branches.
"""
function TransformerBlock(; input_shape::Tuple, heads::Int, hidden_size::Int=4 * first(input_shape),
                          causal::Bool=false, padding_mask::Bool=false, dropout::Real=0,
                          epsilon::Real=1//100000, precision::Type{T}=Float32) where {T<:AbstractFloat}
    0 <= dropout < 1 || throw(ArgumentError("dropout must be in [0, 1)"))
    attention = Attention(; input_shape, heads, causal, padding_mask, precision)
    normalization = LayerNorm(; input_shape, epsilon, precision)
    feedforward = Sequential(
        LayerNorm(; input_shape, epsilon, precision),
        Dense(; input_shape, layer_size=hidden_size, activation_function=GELU, precision),
        Dense(; input_shape=(hidden_size, input_shape[2]), layer_size=input_shape[1],
              activation_function=None, precision);
        precision,
    )
    if dropout > 0
        add_layer!(feedforward, Dropout; rate=dropout)
    end
    residual_feedforward = Residual(feedforward)

    if padding_mask
        shapes = (input_shape, (input_shape[2],))
        nodes = (Node(normalization, 1), Node(attention, 3, 2))
        if dropout > 0
            nodes = (nodes..., Node(Dropout(; input_shape, rate=dropout, precision), 4))
        end
        attended = length(shapes) + length(nodes)
        return Graph(shapes, nodes..., Node(Add(input_shape; precision), 1, attended),
                     Node(residual_feedforward, attended + 1); precision)
    end

    branch = Sequential(normalization, attention; precision)
    if dropout > 0
        add_layer!(branch, Dropout; rate=dropout)
    end
    return Sequential(Residual(branch), residual_feedforward; precision)
end

accepts_precision(::typeof(TransformerBlock)) = true
