"""A pre-norm Transformer block. Pass a key mask as a second input when needed."""
struct TransformerBlock{N,A,D,F} <: AbstractLayer
    norm::N
    attention::A
    dropout::D
    feedforward::F
end

function TransformerBlock(width::Int; heads::Int, hidden_size::Int=4 * width, causal::Bool=false,
                          dropout::Real=0, epsilon::Real=1//100000, precision::Type{T}=Float32) where {T<:AbstractFloat}
    norm = LayerNorm(width; epsilon, precision)
    attention = Attention(width; heads, causal, precision)
    feedforward = Residual(Sequential(
        LayerNorm(width; epsilon, precision),
        Dense(width, hidden_size; activation=GELU(), precision),
        Dense(hidden_size, width; precision),
        Dropout(dropout),
    ))
    return TransformerBlock(norm, attention, Dropout(dropout), feedforward)
end

layer_children(block::TransformerBlock) =
    (; norm=block.norm, attention=block.attention, dropout=block.dropout, feedforward=block.feedforward)
layer_cache(::TransformerBlock) = ResidualCache()
output_shape(block::TransformerBlock, input) = output_shape(block.attention, input)

function forward!(::TransformerBlock, run::Execution, input, mode)
    data = input isa Tuple ? first(input) : input
    norm = execute!(run.children.norm, data, mode)
    attention_input = input isa Tuple ? (norm, Base.tail(input)...) : norm
    attended = execute!(run.children.attention, attention_input, mode)
    branch = execute!(run.children.dropout, attended, mode)
    hidden = sum_tensors!(run.cache, :output, data, branch)
    return execute!(run.children.feedforward, hidden, mode)
end

function backward!(::TransformerBlock, run::Execution, input, delta; compute_parameters)
    hidden = differentiate!(run.children.feedforward, delta; compute_parameters)
    attended = differentiate!(run.children.dropout, hidden; compute_parameters)
    norm = differentiate!(run.children.attention, attended; compute_parameters)
    branch = differentiate!(run.children.norm, input isa Tuple ? first(norm) : norm; compute_parameters)
    dx = sum_tensors!(run.cache, :gradient, hidden, branch)
    return input isa Tuple ? (dx, nothing) : dx
end
