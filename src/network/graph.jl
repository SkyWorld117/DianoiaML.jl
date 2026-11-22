"""
    Node(layer, parents...)

An invocation of a handwritten layer. Indices refer to graph inputs first,
then to earlier nodes. Each invocation has independent forward buffers.
"""
struct Node{L,P<:Tuple}
    layer::L
    parents::P
    Node(layer::L, parents::Int...) where {L} = new{L,typeof(parents)}(layer, parents)
end

"""
    Graph(input_shapes, nodes...; outputs, precision)

A topologically ordered graph. For two inputs, indices 1 and 2 are inputs and
index 3 is the first node. Nodes with several parents receive a tuple. Multiple
selected outputs are returned as a tuple in the order specified.
"""
mutable struct Graph{T<:AbstractFloat,N<:Tuple,S<:Tuple,O<:Tuple} <: AbstractLayer
    nodes::N
    input_shapes::S
    outputs::O
    input_shape::Tuple
    output_shape::Tuple
    values::Vector{Union{Array,Tuple}}
    output::Union{Array,Tuple}
end

function Graph(input_shapes::Tuple{Vararg{Tuple}}, nodes::Node...;
               outputs::Tuple=(length(input_shapes) + length(nodes),),
               precision::Type{T}=isempty(nodes) ? Float32 : model_precision(first(nodes).layer)) where {T<:AbstractFloat}
    ninputs = length(input_shapes)
    ninputs > 0 && !isempty(nodes) || throw(ArgumentError("a graph requires inputs and nodes"))
    total = ninputs + length(nodes)
    !isempty(outputs) && all(i -> i isa Int && 1 <= i <= total, outputs) && allunique(outputs) ||
        throw(ArgumentError("invalid graph output indices"))

    shapes = Tuple[input_shapes...]
    for (i, node) in enumerate(nodes)
        !isempty(node.parents) && all(p -> 1 <= p < ninputs + i, node.parents) ||
            throw(ArgumentError("node parents must be graph inputs or earlier nodes"))
        expected = graph_inputs(shapes, node.parents)
        check_input_shape(node.layer, expected)
        push!(shapes, node.layer.output_shape)
    end

    used = falses(total)
    for i in outputs
        used[i] = true
    end
    for i in length(nodes):-1:1
        used[ninputs + i] || throw(ArgumentError("every graph node must contribute to an output"))
        for p in nodes[i].parents
            used[p] = true
        end
    end

    checked_precision(T)
    graph = Graph{T,typeof(nodes),typeof(input_shapes),typeof(outputs)}(
                  nodes, input_shapes, outputs, graph_inputs(shapes, ntuple(identity, ninputs)),
                  graph_inputs(shapes, outputs), Vector{Union{Array,Tuple}}(undef, total), ())
    validate_layer_tree!(graph)
    return graph
end

precision_type(::Graph{T}) where {T} = T

graph_inputs(values, indices::Tuple{Int}) = values[only(indices)]
graph_inputs(values, indices::Tuple) = map(i -> values[i], indices)

layer_children(graph::Graph) = map(node -> node.layer, graph.nodes)
supports_backward(graph::Graph) = all(supports_backward, layer_children(graph))

function initialize!(graph::Graph, batch::Int)
    for node in graph.nodes
        initialize!(node.layer, batch)
    end
end

activate!(graph::Graph, input; training::Bool=false) = activate_layer(graph, input, training)

function activate_layer(graph::Graph, input, training)
    input = layer_input(graph, input)
    ninputs = length(graph.input_shapes)
    inputs = ninputs == 1 ? (input,) : input
    inputs isa Tuple && length(inputs) == ninputs || throw(DimensionMismatch("graph input count differs"))
    for (i, value) in enumerate(inputs)
        graph.values[i] = value
    end
    for (i, node) in enumerate(graph.nodes)
        graph.values[ninputs + i] = activate_layer(node.layer, graph_inputs(graph.values, node.parents), training)
    end
    return graph.output = graph_inputs(graph.values, graph.outputs)
end

copy_cotangent(value::AbstractArray) = copy(value)
copy_cotangent(value::Tuple) = map(copy_cotangent, value)
copy_cotangent(value::Union{Nothing,LogitGradient}) = value

function merge_cotangents(current, value)
    value === nothing && return current
    current === nothing && return copy_cotangent(value)
    if current isa Tuple && value isa Tuple
        length(current) == length(value) || throw(DimensionMismatch("graph cotangent tuple lengths differ"))
        return map(merge_cotangents, current, value)
    elseif current isa AbstractArray && value isa AbstractArray
        size(current) == size(value) || throw(DimensionMismatch("graph cotangent shapes differ"))
        current .+= value
        return current
    end
    throw(ArgumentError("incompatible graph cotangents"))
end

zero_cotangent(value::Array) = zero(value)
zero_cotangent(value::Tuple) = map(zero_cotangent, value)

function backward!(graph::Graph, input, upstream)
    cotangents = Any[nothing for _ in graph.values]
    outputs = length(graph.outputs) == 1 ? (upstream,) : upstream
    outputs isa Tuple && length(outputs) == length(graph.outputs) ||
        throw(DimensionMismatch("graph output cotangent count differs"))
    for (i, delta) in zip(graph.outputs, outputs)
        cotangents[i] = merge_cotangents(cotangents[i], cotangent_as(graph.values[i], delta))
    end
    ninputs = length(graph.input_shapes)
    for i in length(graph.nodes):-1:1
        node = graph.nodes[i]
        delta = cotangents[ninputs + i]
        delta === nothing && (delta = zero_cotangent(graph.values[ninputs + i]))
        gradients = propagate_layer!(node.layer, graph_inputs(graph.values, node.parents), delta)
        parts = length(node.parents) == 1 ? (gradients,) : gradients
        parts isa Tuple && length(parts) == length(node.parents) ||
            throw(DimensionMismatch("a multi-input layer must return one cotangent per input"))
        for (p, gradient) in zip(node.parents, parts)
            cotangents[p] = merge_cotangents(cotangents[p], gradient)
        end
    end
    return graph_inputs(cotangents, ntuple(identity, ninputs))
end

"""Add two or more equally shaped graph inputs elementwise."""
mutable struct Add{T<:AbstractFloat} <: AbstractLayer
    output_shape::Tuple
    output::Array{T}
    Add(shape::Tuple; precision::Type{T}=Float32) where {T<:AbstractFloat} =
        new{checked_precision(T)}(shape)
end

precision_type(::Add{T}) where {T} = T
initialize!(::Add, ::Int) = nothing
supports_backward(::Add) = true

function activate!(layer::Add{T}, input::Tuple) where {T}
    length(input) >= 2 || throw(ArgumentError("Add requires at least two inputs"))
    all(x -> size(x) == size(first(input)), input) || throw(DimensionMismatch("Add input shapes differ"))
    if !isdefined(layer, :output) || size(layer.output) != size(first(input))
        layer.output = similar(first(input), T)
    end
    copyto!(layer.output, first(input))
    for x in Base.tail(input)
        layer.output .+= x
    end
    return layer.output
end

backward!(::Add, input::Tuple, delta) = map(_ -> delta, input)

"""Select one tensor from a tuple-valued graph node."""
mutable struct Select <: AbstractLayer
    index::Int
    input_shape::Tuple
    output_shape::Tuple
    output::Union{Array,Tuple}

    function Select(index::Int, input_shapes::Tuple{Vararg{Tuple}})
        1 <= index <= length(input_shapes) || throw(BoundsError(input_shapes, index))
        new(index, input_shapes, input_shapes[index], ())
    end
end

initialize!(::Select, ::Int) = nothing
supports_backward(::Select) = true
activate!(layer::Select, input::Tuple) = (layer.output = input[layer.index])
backward!(layer::Select, input::Tuple, delta) = ntuple(i -> i == layer.index ? delta : nothing, length(input))

"""Create another Dense invocation with the same trainable arrays and independent buffers."""
function share_parameters(layer::Dense)
    shared = Dense(input_shape=layer.input_shape, layer_size=layer.layer_size,
                   activation_function=layer.activation_function, randomization=false,
                   precision=model_precision(layer))
    shared.weights = layer.weights
    shared.biases = layer.biases
    return shared
end

function share_parameters(layer::Union{Conv2D,TConv2D})
    constructor = layer isa Conv2D ? Conv2D : TConv2D
    shared = constructor(input_shape=layer.input_shape, filter=layer.filter, padding=layer.padding,
               kernel_size=layer.kernel_size, strides=layer.strides,
               activation_function=layer.activation_function, randomization=false,
               precision=model_precision(layer))
    shared.filters = layer.filters
    shared.biases = layer.biases
    return shared
end
