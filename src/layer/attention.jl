module AttentionM
    using LinearAlgebra: mul!
    using ..DianoiaML: AbstractLayer, Dense, None, checked_precision, accumulation_type, float_input
    import ..DianoiaML
    import ..DianoiaML: precision_type, initialize!, activate!, backward!, update!, supports_backward,
                        layer_children, save_layer, load_layer!

    struct AttentionWorkspace{T,A}
        probabilities::Array{A,4}
        score_gradient::Matrix{A}
        query::Union{Array{A,3},Nothing}
        key::Union{Array{A,3},Nothing}
        value::Union{Array{A,3},Nothing}
        attended::Array{A,3}
        projection_input::Array{T,3}
        query_gradient::Array{A,3}
        key_gradient::Array{A,3}
        value_gradient::Array{A,3}
    end

    function AttentionWorkspace(::Type{T}, ::Type{A}, query_shape, key_shape, heads) where {T,A}
        queries, keys, batch = query_shape[2], key_shape[2], query_shape[3]
        attended = Array{A}(undef, query_shape)
        return AttentionWorkspace{T,A}(
            Array{A}(undef, keys, queries, heads, batch), Matrix{A}(undef, keys, queries),
            T === A ? nothing : similar(attended),
            T === A ? nothing : Array{A}(undef, key_shape),
            T === A ? nothing : Array{A}(undef, key_shape),
            attended, T === A ? attended : similar(attended, T),
            similar(attended), Array{A}(undef, key_shape), Array{A}(undef, key_shape))
    end

    """
        Attention(; input_shape, heads, context_shape=nothing, causal=false,
                    padding_mask=false, precision=Float32)

    Multi-head scaled dot-product attention on features × tokens × examples.
    Supply a query tensor alone for self-attention, or `(query, context)` when
    `context_shape` is given. With `padding_mask=true`, append a Boolean
    keys × examples tensor to the inputs. True entries allow that key.
    `causal=true` restricts self-attention to the current and earlier tokens.
    """
    mutable struct Attention{T<:AbstractFloat,A<:AbstractFloat,S<:Tuple} <: AbstractLayer
        input_shape::S
        output_shape::NTuple{2,Int}
        heads::Int
        causal::Bool
        cross_attention::Bool
        padding_mask::Bool
        scale::A
        query::Dense{T,A}
        key::Dense{T,A}
        value::Dense{T,A}
        projection::Dense{T,A}
        workspace::AttentionWorkspace{T,A}
        output::Array{T,3}
        δ::Array{T,3}

        function Attention(; input_shape::Tuple, heads::Int, context_shape::Union{Tuple,Nothing}=nothing,
                           causal::Bool=false, padding_mask::Bool=false, randomization::Bool=true,
                           precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            key_shape = something(context_shape, input_shape)
            for shape in (input_shape, key_shape)
                shape isa NTuple{2,Int} && all(>(0), shape) ||
                    throw(ArgumentError("Attention shapes must be (features, tokens) with positive dimensions"))
            end
            width = input_shape[1]
            heads > 0 && width % heads == 0 ||
                throw(ArgumentError("attention heads must divide the query feature count"))
            cross = context_shape !== nothing
            causal && cross && throw(ArgumentError("causal attention requires self-attention"))
            shapes = cross ? (input_shape, key_shape) : (input_shape,)
            padding_mask && (shapes = (shapes..., (key_shape[2],)))
            shape = length(shapes) == 1 ? only(shapes) : shapes
            A = accumulation_type(T)
            project(features) = Dense(; input_shape=features, layer_size=width,
                                      activation_function=None, randomization, precision=T)
            new{T,A,typeof(shape)}(shape, input_shape, heads, causal, cross, padding_mask,
                inv(sqrt(A(width ÷ heads))), project(input_shape), project(key_shape),
                project(key_shape), project(input_shape))
        end
    end

    precision_type(::Attention{T}) where {T} = T
    supports_backward(::Attention) = true
    layer_children(layer::Attention) = (layer.query, layer.key, layer.value, layer.projection)

    function initialize!(layer::Attention, batch::Int)
        foreach(projection -> initialize!(projection, batch), layer_children(layer))
        layer.output = layer.projection.output
        layer.δ = layer.query.δ
        return
    end

    function attention_inputs(layer::Attention{T}, input) where {T}
        parts = input isa Tuple ? input : (input,)
        length(parts) == 1 + layer.cross_attention + layer.padding_mask ||
            throw(DimensionMismatch("attention input count differs"))
        query = first(parts)
        context = layer.cross_attention ? parts[2] : query
        query isa Array{T,3} && context isa Array{T,3} ||
            throw(ArgumentError("attention requires features × tokens × examples tensors of type $T"))
        size(query, 1) == size(layer.query.weights, 2) && size(context, 1) == size(layer.key.weights, 2) ||
            throw(DimensionMismatch("attention input feature count differs"))
        size(query, 2) > 0 && size(context, 2) > 0 && size(query, 3) == size(context, 3) ||
            throw(DimensionMismatch("attention needs nonempty sequences with matching batch sizes"))
        mask = layer.padding_mask ? last(parts) : nothing
        if layer.padding_mask
            mask isa AbstractMatrix{Bool} || throw(ArgumentError("attention padding mask must be Boolean"))
            size(mask) == (size(context, 2), size(context, 3)) ||
                throw(DimensionMismatch("attention padding mask must have shape (keys, examples)"))
        end
        return query, context, mask
    end

    from_projection(buffer::Nothing, values) = values
    from_projection(buffer::Array, values) = copyto!(buffer, values)

    function normalize_scores!(scores::AbstractMatrix{A}, causal, mask, example) where {A}
        @inbounds for query in axes(scores, 2)
            last_key = causal ? query : size(scores, 1)
            peak = A(-Inf)
            for key in axes(scores, 1)
                visible = key <= last_key && (mask === nothing || mask[key, example])
                scores[key, query] = visible ? scores[key, query] : A(-Inf)
                peak = max(peak, scores[key, query])
            end
            if peak == -Inf
                # An example can have no valid keys, for instance when it is all padding.
                for key in axes(scores, 1)
                    scores[key, query] = zero(A)
                end
                continue
            end
            total = zero(A)
            for key in axes(scores, 1)
                weight = exp(scores[key, query] - peak)
                scores[key, query] = weight
                total += weight
            end
            @simd for key in axes(scores, 1)
                scores[key, query] /= total
            end
        end
    end

    function activate!(layer::Attention{T,A}, input) where {T,A}
        query, context, mask = attention_inputs(layer, input)
        q = activate!(layer.query, query)
        k = activate!(layer.key, context)
        v = activate!(layer.value, context)
        dims = (size(k, 2), size(q, 2), layer.heads, size(q, 3))
        if !isdefined(layer, :workspace) || size(layer.workspace.probabilities) != dims
            layer.workspace = AttentionWorkspace(T, A, size(q), size(k), layer.heads)
        end
        work = layer.workspace
        q = from_projection(work.query, q)
        k = from_projection(work.key, k)
        v = from_projection(work.value, v)
        head_width = size(q, 1) ÷ layer.heads
        @views for example in axes(q, 3), head in 1:layer.heads
            features = (head - 1) * head_width + 1:head * head_width
            probabilities = work.probabilities[:, :, head, example]
            mul!(probabilities, transpose(k[features, :, example]), q[features, :, example],
                 layer.scale, zero(A))
            normalize_scores!(probabilities, layer.causal, mask, example)
            mul!(work.attended[features, :, example], v[features, :, example], probabilities)
        end
        T === A || copyto!(work.projection_input, work.attended)
        layer.δ = layer.query.δ
        return layer.output = activate!(layer.projection, work.projection_input)
    end

    function softmax_gradient!(gradient, probabilities)
        @inbounds for query in axes(probabilities, 2)
            projection = zero(eltype(gradient))
            @simd for key in axes(probabilities, 1)
                projection += probabilities[key, query] * gradient[key, query]
            end
            @simd for key in axes(probabilities, 1)
                gradient[key, query] = probabilities[key, query] * (gradient[key, query] - projection)
            end
        end
    end

    function backward!(layer::Attention{T,A}, input, delta::Array{T,3}; compute_parameters=true) where {T,A}
        query, context, _ = attention_inputs(layer, input)
        work = layer.workspace
        d_attended = backward!(layer.projection, work.projection_input, delta; compute_parameters)
        T === A || (d_attended = copyto!(work.attended, d_attended))
        q = something(work.query, layer.query.output)
        k = something(work.key, layer.key.output)
        v = something(work.value, layer.value.output)
        head_width = size(q, 1) ÷ layer.heads
        @views for example in axes(q, 3), head in 1:layer.heads
            features = (head - 1) * head_width + 1:head * head_width
            probabilities = work.probabilities[:, :, head, example]
            upstream = d_attended[features, :, example]
            ds = work.score_gradient
            mul!(work.value_gradient[features, :, example], upstream, transpose(probabilities))
            mul!(ds, transpose(v[features, :, example]), upstream)
            softmax_gradient!(ds, probabilities)
            mul!(work.query_gradient[features, :, example], k[features, :, example],
                 ds, layer.scale, zero(A))
            mul!(work.key_gradient[features, :, example], q[features, :, example],
                 transpose(ds), layer.scale, zero(A))
        end
        dq = backward!(layer.query, query, float_input(T, work.query_gradient); compute_parameters)
        dk = backward!(layer.key, context, float_input(T, work.key_gradient); compute_parameters)
        dv = backward!(layer.value, context, float_input(T, work.value_gradient); compute_parameters)
        if layer.cross_attention
            dk .+= dv
            return layer.padding_mask ? (dq, dk, nothing) : (dq, dk)
        end
        dq .= convert.(A, dq) .+ convert.(A, dk) .+ convert.(A, dv)
        return layer.padding_mask ? (dq, nothing) : dq
    end

    function update!(layer::Attention, optimizer::String, input, delta, rate::Real, settings::Tuple, direction::Int=1)
        derivative = backward!(layer, input, delta; compute_parameters=direction != 0)
        direction == 0 || DianoiaML.legacy_step!((layer,), 1:1, optimizer, rate, settings, direction)
        return derivative
    end

    function save_layer(layer::Attention, file, id::String)
        write(file, id, "Attention")
        write(file, id * "input_shape", collect(layer.output_shape))
        write(file, id * "heads", layer.heads)
        write(file, id * "causal", layer.causal)
        write(file, id * "padding_mask", layer.padding_mask)
        layer.cross_attention && write(file, id * "context_shape", collect(layer.key.input_shape))
        for (i, projection) in enumerate(layer_children(layer))
            save_layer(projection, file, id * "projection" * string(i))
        end
    end

    function get_args(file, id::String)
        context_key = id * "context_shape"
        context_shape = haskey(file, context_key) ? Tuple(read(file, context_key)) : nothing
        return (; input_shape=Tuple(read(file, id * "input_shape")), context_shape,
                heads=Int(read(file, id * "heads")), causal=Bool(read(file, id * "causal")),
                padding_mask=Bool(read(file, id * "padding_mask")), randomization=false)
    end

    function load_layer!(layer::Attention, file, id::String)
        for (i, projection) in enumerate(layer_children(layer))
            load_layer!(projection, file, id * "projection" * string(i))
        end
    end
end
