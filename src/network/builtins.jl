function resnet_block(input, output, stride, precision)
    branch = Sequential(
        Conv2D(input, output; kernel_size=(3, 3), stride, padding=1, precision),
        BatchNorm2D(output; precision),
        ReLU(),
        Conv2D(output, output; kernel_size=(3, 3), padding=1, precision),
        BatchNorm2D(output; precision),
    )
    if input == output && stride == (1, 1)
        return Sequential(Residual(branch), ReLU())
    end
    projection = Sequential(
        Conv2D(input, output; kernel_size=(1, 1), stride, precision),
        BatchNorm2D(output; precision),
    )
    return Graph(; inputs=(:x,), nodes=(
        branch=Node(branch, :x),
        shortcut=Node(projection, :x),
        sum=Node(Add(), :branch, :shortcut),
        activation=Node(ReLU(), :sum),
    ))
end

"""A compact ResNet-18 image classifier returning class logits."""
function ResNet18(input_channels::Int, classes::Int; width::Int=32, precision::Type{T}=Float32) where {T<:AbstractFloat}
    input_channels > 0 && classes > 0 && width > 0 ||
        throw(ArgumentError("ResNet18 requires positive dimensions"))
    stem = Sequential(
        Conv2D(input_channels, width; kernel_size=(7, 7), stride=(2, 2), padding=3, precision),
        BatchNorm2D(width; precision),
        ReLU(),
        MaxPooling2D((2, 2)),
    )
    stages = ntuple(4) do stage
        output = width * 2^(stage - 1)
        input = stage == 1 ? width : output ÷ 2
        stride = stage == 1 ? (1, 1) : (2, 2)
        return Sequential(resnet_block(input, output, stride, precision),
                          resnet_block(output, output, (1, 1), precision))
    end
    return Sequential(stem, stages..., GlobalMeanPool2D(), Dense(8 * width, classes; precision);
                      input_shape=(nothing, nothing, input_channels))
end

unet_block(input, output, precision) = Sequential(
    Conv2D(input, output; kernel_size=(3, 3), padding=1, activation=ReLU(), precision),
    Conv2D(output, output; kernel_size=(3, 3), padding=1, activation=ReLU(), precision),
)

"""A same-padded U-Net producing classes × width × height × batch logits."""
function UNet(input_channels::Int, classes::Int; channels::Tuple{Vararg{Int}}=(16, 32, 64, 128),
              precision::Type{T}=Float32) where {T<:AbstractFloat}
    input_channels > 0 && classes > 0 && length(channels) >= 2 && all(>(0), channels) ||
        throw(ArgumentError("UNet requires positive channels and classes and at least two levels"))
    nodes = Pair{Symbol,Node}[]
    source = :image
    for i in eachindex(channels)
        encoder = Symbol(:encoder_, i)
        input = i == 1 ? input_channels : channels[i - 1]
        push!(nodes, encoder => Node(unet_block(input, channels[i], precision), source))
        source = encoder
        if i < length(channels)
            pool = Symbol(:pool_, i)
            push!(nodes, pool => Node(MaxPooling2D((2, 2)), encoder))
            source = pool
        end
    end
    for i in length(channels)-1:-1:1
        up, joined, decoder = Symbol(:up_, i), Symbol(:join_, i), Symbol(:decoder_, i)
        project = Sequential(UpSampling2D((2, 2)),
                            Conv2D(channels[i + 1], channels[i]; kernel_size=(1, 1), precision))
        push!(nodes, up => Node(project, source),
              joined => Node(Concatenate(3), Symbol(:encoder_, i), up),
              decoder => Node(unet_block(2 * channels[i], channels[i], precision), joined))
        source = decoder
    end
    head = Sequential(Conv2D(first(channels), classes; kernel_size=(1, 1), precision), Permute((3, 1, 2)))
    push!(nodes, :logits => Node(head, source))
    return Graph(; inputs=(:image,), nodes=(; nodes...),
                 input_shapes=(; image=(nothing, nothing, input_channels)))
end

"""A compact GPT-style causal decoder accepting tokens × batch integer IDs."""
function GPT(vocabulary::Int; context_length::Int=128, width::Int=128, depth::Int=4,
             heads::Int=4, dropout::Real=0, precision::Type{T}=Float32) where {T<:AbstractFloat}
    depth > 0 || throw(ArgumentError("GPT requires at least one Transformer block"))
    blocks = ntuple(_ -> TransformerBlock(width; heads, causal=true, dropout, precision), depth)
    return Sequential(
        Embedding(vocabulary, width; precision),
        PositionalEmbedding(width, context_length; precision),
        Dropout(dropout),
        blocks...,
        LayerNorm(width; precision),
        Dense(width, vocabulary; precision);
        input_shape=(nothing,),
    )
end

nbeats_projection(width, basis, output, precision) =
    Sequential(Dense(width, basis; precision), Dense(basis, output; precision))

"""A generic N-BEATS forecaster for univariate lookback windows."""
function NBEATS(lookback::Int, horizon::Int; width::Int=256, blocks::Int=4, hidden_layers::Int=4,
                basis::Int=32, precision::Type{T}=Float32) where {T<:AbstractFloat}
    all(>(0), (lookback, horizon, width, blocks, hidden_layers, basis)) ||
        throw(ArgumentError("NBEATS requires positive dimensions"))
    nodes = Pair{Symbol,Node}[]
    forecasts = Symbol[]
    source = :history
    for i in 1:blocks
        hidden, forecast = Symbol(:hidden_, i), Symbol(:forecast_, i)
        trunk = Sequential(Dense(lookback, width; activation=ReLU(), precision),
                           ntuple(_ -> Dense(width, width; activation=ReLU(), precision), hidden_layers - 1)...)
        push!(nodes, hidden => Node(trunk, source),
              forecast => Node(nbeats_projection(width, basis, horizon, precision), hidden))
        push!(forecasts, forecast)
        if i < blocks
            backcast, residual = Symbol(:backcast_, i), Symbol(:residual_, i)
            push!(nodes, backcast => Node(nbeats_projection(width, basis, lookback, precision), hidden),
                  residual => Node(Subtract(), source, backcast))
            source = residual
        end
    end
    if blocks > 1
        push!(nodes, :forecast => Node(Add(), forecasts...))
    end
    return Graph(; inputs=(:history,), nodes=(; nodes...), input_shapes=(; history=(lookback,)))
end
