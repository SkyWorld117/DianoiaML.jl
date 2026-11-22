using HDF5: create_group

function model_arrays(model)
    arrays = Dict{String,Array}()
    walk_layers(model) do layer, prefix
        for (name, (value, _)) in pairs(parameter_gradients(layer))
            arrays[prefix * String(name)] = value
        end
        for (name, value) in pairs(persistent_buffers(layer))
            arrays[prefix * "buffers." * String(name)] = value
        end
    end
    return arrays
end

"""Return a snapshot of the model's named parameters and persistent buffers."""
state_dict(model) = Dict(name => copy(value) for (name, value) in model_arrays(model))

function validate_state_arrays(dest, saved)
    keys(dest) == keys(saved) || throw(ArgumentError("checkpoint model keys differ"))
    sources = IdDict{Any,Array}()
    for (name, value) in dest
        source = saved[name]
        source isa Array && size(value) == size(source) && eltype(value) == eltype(source) ||
            throw(DimensionMismatch("invalid checkpoint array at $name"))
        previous = get!(sources, value, source)
        isequal(previous, source) || throw(ArgumentError("conflicting checkpoint values for tied array $name"))
    end
end

"""Load parameters and buffers into an existing model, preserving array identity."""
function load_state!(model, saved)
    dest = model_arrays(model)
    validate_state_arrays(dest, saved)
    for (name, value) in dest
        copyto!(value, saved[name])
    end
    return model
end

function checkpoint_aliases(state)
    return [(name, state.index[value]) for (name, value, _) in parameter_occurrences(state.model)]
end

"""
    save_checkpoint(path, state)

Save model arrays, optimizer moments and steps, parameter groups, pending
gradient accumulation, epoch count, and the training state's dropout stream.
The caller owns the architecture and any state in its batch iterator.
"""
function save_checkpoint(path::AbstractString, state::TrainingState)
    state.input === nothing || throw(ArgumentError("complete backward! before saving a checkpoint"))
    check_parameter_bindings(state)
    aliases = checkpoint_aliases(state)
    arrays = model_arrays(state.model)
    destination = abspath(path)
    mktemp(dirname(destination)) do temporary, io
        close(io)
        h5open(temporary, "w") do file
            write(file, "version", 1)
            write(file, "optimizer", state.optimizer.kind)
            write(file, "settings", [state.optimizer.α, state.optimizer.beta1, state.optimizer.beta2, state.optimizer.epsilon])
            write(file, "progress", [state.samples, state.microbatches, state.epochs])
            write(file, "random", [state.seed, state.forwards])
            write(file, "array_count", length(arrays))
            write(file, "parameter_count", length(state.slots))
            write(file, "alias_count", length(aliases))

            model = create_group(file, "model")
            for (i, name) in enumerate(sort!(collect(keys(arrays))))
                write(model, "name_$i", name)
                write(model, "value_$i", arrays[name])
            end
            optimizer = create_group(file, "parameters")
            for (i, slot) in enumerate(state.slots)
                group = create_group(optimizer, string(i))
                write(group, "name", state.names[i])
                write(group, "rate", slot.α)
                write(group, "trainable", slot.trainable)
                write(group, "step", slot.step)
                write(group, "gradient", slot.gradient)
                if slot.first !== nothing
                    write(group, "first", slot.first)
                    write(group, "second", slot.second)
                end
            end
            links = create_group(file, "aliases")
            for (i, (name, index)) in enumerate(aliases)
                write(links, "name_$i", name)
                write(links, "index_$i", index)
            end
        end
        Base.Filesystem.rename(temporary, destination)
    end
    return path
end

function read_slot_array!(dest, group, key)
    value = read(group, key)
    size(dest) == size(value) && eltype(dest) == eltype(value) ||
        throw(DimensionMismatch("checkpoint optimizer array $key differs"))
    copyto!(dest, value)
end

"""Restore a training checkpoint into a state built around the same architecture and parameter ties."""
function load_checkpoint!(state::TrainingState, path::AbstractString)
    state.input === nothing || throw(ArgumentError("complete backward! before loading a checkpoint"))
    saved, restored = h5open(path, "r") do file
        read(file, "version") == 1 || throw(ArgumentError("unsupported checkpoint version"))
        α, beta1, beta2, epsilon = read(file, "settings")
        optimizer = Optimizer(Symbol(read(file, "optimizer")); α, beta1, beta2, epsilon)
        seed, forwards = read(file, "random")
        restored = TrainingState(state.model, optimizer; seed)
        restored.forwards = forwards
        restored.samples, restored.microbatches, restored.epochs = read(file, "progress")
        0 <= restored.microbatches <= restored.samples && restored.epochs >= 0 &&
            (restored.samples == 0) == (restored.microbatches == 0) ||
            throw(ArgumentError("invalid checkpoint progress"))

        links = file["aliases"]
        aliases = [(read(links, "name_$i"), read(links, "index_$i"))
                   for i in 1:read(file, "alias_count")]
        aliases == checkpoint_aliases(restored) ||
            throw(ArgumentError("checkpoint parameter names or ties differ"))
        length(restored.slots) == read(file, "parameter_count") ||
            throw(ArgumentError("checkpoint parameter count differs"))
        for (i, slot) in enumerate(restored.slots)
            group = file["parameters/$i"]
            read(group, "name") == restored.names[i] || throw(ArgumentError("checkpoint parameter order differs"))
            slot.α = read(group, "rate")
            validate_rate(slot.α)
            slot.trainable = read(group, "trainable")
            slot.step = read(group, "step")
            slot.step >= 0 || throw(ArgumentError("invalid checkpoint optimizer step"))
            read_slot_array!(slot.gradient, group, "gradient")
            if slot.first !== nothing
                read_slot_array!(slot.first, group, "first")
                read_slot_array!(slot.second, group, "second")
            end
        end
        model = file["model"]
        saved = Dict(read(model, "name_$i") => read(model, "value_$i") for i in 1:read(file, "array_count"))
        return saved, restored
    end

    load_state!(state.model, saved)
    state.optimizer = restored.optimizer
    state.slots = restored.slots
    state.names = restored.names
    state.index = restored.index
    state.sources = restored.sources
    state.seed = restored.seed
    state.forwards = restored.forwards
    state.samples = restored.samples
    state.microbatches = restored.microbatches
    state.epochs = restored.epochs
    state.batch = 0
    return state
end
