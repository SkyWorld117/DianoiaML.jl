function to_device(backend::KA.Backend, model::Union{AbstractLayer,GAN})
    copies = IdDict{Any,AbstractArray}()
    move(array) = get!(() -> to_device(backend, array), copies, array)
    return adapt_model(move, model)
end

function adapt_model(move, layer::AbstractLayer)
    isempty(layer_parameters(layer)) && isempty(persistent_buffers(layer)) && isempty(layer_children(layer)) ||
        throw(ArgumentError("define DianoiaML.adapt_model for $(typeof(layer))"))
    return layer
end

adapt_model(move, layer::Dense) = Dense(move(layer.weights), move(layer.bias), layer.activation)

adapt_model(move, layer::Convolution{P}) where {P} =
    Convolution{P}(move(layer.filters), move(layer.bias), layer.padding, layer.stride, layer.activation)

adapt_model(move, layer::LayerNorm) = LayerNorm(move(layer.scale), move(layer.bias), layer.epsilon)

adapt_model(move, layer::Attention) =
    Attention(layer.heads, layer.causal, layer.cross_attention, layer.scale,
              map(child -> adapt_model(move, child), (layer.query, layer.key, layer.value, layer.projection))...)

adapt_model(move, model::Sequential) =
    Sequential(map(layer -> adapt_model(move, layer), model.layers)...; input_shape=model.input)

adapt_model(move, layer::Residual) = Residual(adapt_model(move, layer.branch))

function adapt_model(move, graph::Graph)
    nodes = map(node -> Node(adapt_model(move, node.layer), node.parents...), graph.nodes)
    return Graph(graph.inputs, nodes, graph.routes, graph.outputs, graph.shapes)
end

adapt_model(move, layer::TransformerBlock) =
    TransformerBlock(adapt_model(move, layer.norm), adapt_model(move, layer.attention),
                     layer.dropout, adapt_model(move, layer.feedforward))

adapt_model(move, model::GAN) =
    GAN(adapt_model(move, model.generator), adapt_model(move, model.discriminator))
