"""Call a layer with inputs from named graph inputs or earlier named nodes."""
struct Node{L<:AbstractLayer,P<:Tuple}
    layer::L
    parents::P
    Node(layer::L, parents::Symbol...) where {L<:AbstractLayer} = new{L,typeof(parents)}(layer, parents)
end

struct Graph{I<:Tuple,N<:NamedTuple,R<:Tuple,O<:Tuple,S<:Tuple} <: AbstractLayer
    inputs::I
    nodes::N
    routes::R
    outputs::O
    shapes::S
end

"""
    Graph(; inputs, nodes, outputs=(last(keys(nodes)),), input_shapes=(;))

Supply input names, a named tuple of nodes in dependency order, and the names
to return. Pass tuple inputs in declaration order or use a named tuple.
"""
function Graph(; inputs::Tuple{Vararg{Symbol}}, nodes::NamedTuple,
               outputs::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}=nothing,
               input_shapes::NamedTuple=(;))
    !isempty(inputs) && !isempty(nodes) || throw(ArgumentError("a graph requires inputs and nodes"))
    names = (inputs..., keys(nodes)...)
    allunique(names) || throw(ArgumentError("graph names must be unique"))
    all(in(inputs), keys(input_shapes)) || throw(ArgumentError("unknown graph input shape"))
    outputs === nothing && (outputs = last(keys(nodes)))
    outputs isa Symbol && (outputs = (outputs,))
    !isempty(outputs) && allunique(outputs) || throw(ArgumentError("graph output names must be unique"))
    indices = Dict(name => i for (i, name) in enumerate(names))
    routes = map(Tuple(nodes), Tuple(length(inputs) .+ (1:length(nodes)))) do node, index
        node isa Node || throw(ArgumentError("expected a Node"))
        isempty(node.parents) && throw(ArgumentError("a node needs at least one parent"))
        return map(node.parents) do parent
            p = get(indices, parent, 0)
            1 <= p < index || throw(ArgumentError("parent $parent must be an input or earlier node"))
            p
        end
    end
    selected = map(outputs) do name
        index = get(indices, name, 0)
        index > 0 || throw(ArgumentError("unknown graph output $name"))
        index
    end
    used = falses(length(names))
    for index in selected
        used[index] = true
    end
    for i in length(routes):-1:1
        used[length(inputs) + i] || throw(ArgumentError("unused graph node $(keys(nodes)[i])"))
        for parent in routes[i]
            used[parent] = true
        end
    end
    graph = Graph(inputs, nodes, routes, selected, map(name -> get(input_shapes, name, nothing), inputs))
    shape(graph)
    return graph
end

layer_children(graph::Graph) = map(node -> node.layer, graph.nodes)
select_values(values, indices::Tuple{Int}) = values[only(indices)]
select_values(values, indices::Tuple) = map(i -> values[i], indices)

input_shape(graph::Graph) = length(graph.inputs) == 1 ? only(graph.shapes) : graph.shapes

function output_shape(graph::Graph, input)
    inputs = length(graph.inputs) == 1 ? (input,) : something(input, graph.shapes)
    length(inputs) == length(graph.inputs) || throw(DimensionMismatch("graph input count differs"))
    shapes = Any[inputs...]
    for (node, route) in zip(graph.nodes, graph.routes)
        push!(shapes, shape(node.layer, select_values(shapes, route)))
    end
    return select_values(shapes, graph.outputs)
end

struct GraphCache
    values::Vector{Tensor}
    gradients::Vector{Union{Nothing,Tensor}}
    scratch::Vector{Union{Nothing,AbstractArray}}
end

function layer_cache(graph::Graph)
    n = length(graph.inputs) + length(graph.nodes)
    return GraphCache(Vector{Tensor}(undef, n), fill!(Vector{Union{Nothing,Tensor}}(undef, n), nothing),
                      fill!(Vector{Union{Nothing,AbstractArray}}(undef, n), nothing))
end

function graph_arguments(graph, data)
    if data isa NamedTuple
        length(data) == length(graph.inputs) && all(haskey(data, name) for name in graph.inputs) ||
            throw(DimensionMismatch("graph input names differ from $(graph.inputs)"))
        return map(name -> getproperty(data, name), graph.inputs)
    end
    length(graph.inputs) == 1 && !(data isa Tuple) && return (data,)
    data isa Tuple && length(data) == length(graph.inputs) || throw(DimensionMismatch("graph inputs differ"))
    return data
end

function forward!(graph::Graph, run::Execution, input, mode)
    args = graph_arguments(graph, input)
    values = run.cache.values
    for (i, value) in enumerate(args)
        values[i] = value
    end
    for (i, child) in enumerate(run.children)
        values[length(args) + i] = execute!(child, select_values(values, graph.routes[i]), mode)
    end
    return select_values(values, graph.outputs)
end

copy_cotangent(::Nothing) = nothing
copy_cotangent(value::AbstractArray) = recording_writes(value) ? copy_tensor!(similar(value), value) : copy(value)
copy_cotangent(value::Union{Tuple,NamedTuple}) = map(copy_cotangent, value)

merge_cotangents(::Nothing, ::Nothing) = nothing
merge_cotangents(current, ::Nothing) = current
merge_cotangents(::Nothing, value) = copy_cotangent(value)
merge_cotangents(current::Tuple, value::Tuple) = map(merge_cotangents, current, value)
merge_cotangents(current::NamedTuple, value::NamedTuple) = map(merge_cotangents, current, value)
function merge_cotangents(current::AbstractArray, value::AbstractArray)
    size(current) == size(value) || throw(DimensionMismatch("graph cotangent shapes differ"))
    return broadcast_tensor!(+, current, current, value)
end

zero_cotangent(value::AbstractArray) = recording_writes(value) ? zeros_like(value, eltype(value), size(value)) : zero(value)
zero_cotangent(value::Union{Tuple,NamedTuple}) = map(zero_cotangent, value)

function graph_cotangent!(cache, i, value, offset)
    current = cache.gradients[i]
    if i > offset && current === nothing && value isa AbstractArray
        buffer = cache.scratch[i]
        matching_buffer(buffer, value) || (buffer = cache.scratch[i] = similar(value))
        cache.gradients[i] = copy_tensor!(buffer, value)
    else
        cache.gradients[i] = merge_cotangents(current, value)
    end
    return
end

function backward!(graph::Graph, run::Execution, input, upstream; compute_parameters)
    gradients = run.cache.gradients
    fill!(gradients, nothing)
    offset = length(graph.inputs)
    selected = length(graph.outputs) == 1 ? (upstream,) : upstream
    for (i, delta) in zip(graph.outputs, selected)
        graph_cotangent!(run.cache, i, cotangent_as(run.cache.values[i], delta), offset)
    end
    for i in length(graph.nodes):-1:1
        delta = gradients[offset + i]
        delta === nothing && (delta = zero_cotangent(run.cache.values[offset + i]))
        source = differentiate!(run.children[i], delta; compute_parameters)
        parents = graph.routes[i]
        parts = length(parents) == 1 ? (source,) : source
        parts isa Union{Tuple,NamedTuple} && length(parts) == length(parents) ||
            throw(DimensionMismatch("a multi-input layer must return one cotangent per input"))
        for (p, gradient) in zip(parents, parts)
            graph_cotangent!(run.cache, p, gradient, offset)
        end
    end
    input isa NamedTuple && return NamedTuple{keys(input)}(map(name -> gradients[findfirst(==(name), graph.inputs)], keys(input)))
    input isa Tuple && return ntuple(i -> gradients[i], offset)
    return gradients[1]
end

struct Add <: AbstractLayer end
layer_cache(::Add) = ActivationCache()

function output_shape(::Add, inputs)
    inputs === nothing && return nothing
    known = filter(!isnothing, inputs)
    isempty(known) && return nothing
    for input in known
        check_shape(first(known), input)
    end
    return first(known)
end

function forward!(::Add, cache::ActivationCache, inputs::Tuple, mode)
    length(inputs) >= 2 && all(x -> size(x) == size(first(inputs)), inputs) ||
        throw(DimensionMismatch("Add requires at least two equally shaped inputs"))
    T = promote_type(map(eltype, inputs)...)
    if !isdefined(cache, :output) || size(cache.output) != size(first(inputs)) ||
       eltype(cache.output) !== T || !same_device(cache.output, first(inputs))
        cache.output = similar(first(inputs), T)
    end
    copy_tensor!(cache.output, first(inputs))
    for input in Base.tail(inputs)
        broadcast_tensor!(+, cache.output, cache.output, input)
    end
    return cache.output
end

backward!(::Add, cache::ActivationCache, inputs, delta; compute_parameters=true) = map(_ -> delta, inputs)

struct Select{K<:Union{Int,Symbol}} <: AbstractLayer
    key::K
end

output_shape(layer::Select, input) = input === nothing ? nothing : input[layer.key]
forward!(layer::Select, ::Nothing, input, mode) = input[layer.key]
function backward!(layer::Select, ::Nothing, input::Tuple, delta; compute_parameters=true)
    return ntuple(i -> i == layer.key ? delta : nothing, length(input))
end

function backward!(layer::Select, ::Nothing, input::NamedTuple, delta; compute_parameters=true)
    selected = layer.key isa Int ? keys(input)[layer.key] : layer.key
    return NamedTuple{keys(input)}(map(name -> name == selected ? delta : nothing, keys(input)))
end
