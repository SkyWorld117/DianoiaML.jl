function model_arrays(model)
    arrays = Dict{String,Array}(parameters(model))
    walk_layers(model) do layer, prefix
        for (name, value) in pairs(persistent_buffers(layer))
            arrays[prefix * "buffers." * string(name)] = value
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
        isequal(previous, source) || throw(ArgumentError("conflicting values for tied array $name"))
    end
end

"""Load parameters and buffers into a model, preserving array identity."""
function load_state!(model, saved)
    dest = model_arrays(model)
    validate_state_arrays(dest, saved)
    for (name, value) in dest
        copyto!(value, saved[name])
    end
    return model
end

settings(optimizer::SGD) = [optimizer.lr]
settings(opt::Union{Adam,AdaBelief}) = [opt.lr, opt.beta1, opt.beta2, opt.epsilon]

function read_optimizer(file)
    kind = read(file, "optimizer")
    values = read_array(file, "settings")
    kind == "SGD" && return SGD(; lr=only(values))
    constructor = kind == "Adam" ? Adam : kind == "AdaBelief" ? AdaBelief :
                  throw(ArgumentError("unknown checkpoint optimizer $kind"))
    lr, beta1, beta2, epsilon = values
    return constructor(; lr, beta1, beta2, epsilon)
end

function write_training_state(file, state::TrainingState)
    state.pending && throw(ArgumentError("complete backward! before saving a checkpoint"))
    arrays = model_arrays(state.model)
    write(file, "optimizer", string(nameof(typeof(state.optimizer))))
    write_array(file, "settings", settings(state.optimizer))
    write(file, "progress", [state.samples, state.microbatches, state.epochs])
    write(file, "random", [state.seed, state.forwards])
    write(file, "array_count", length(arrays))
    write(file, "parameter_count", length(state.slots))
    write(file, "alias_count", length(state.aliases))

    model = create_group(file, "model")
    for (i, name) in enumerate(sort!(collect(keys(arrays))))
        write(model, "name_$i", name)
        write_array(model, "value_$i", arrays[name])
    end
    optimizer = create_group(file, "parameters")
    for (i, slot) in enumerate(state.slots)
        group = create_group(optimizer, string(i))
        write(group, "name", state.names[i])
        write_number(group, "rate", slot.lr)
        write(group, "trainable", slot.trainable)
        write(group, "step", slot.step)
        write_array(group, "gradient", slot.gradient)
        if slot.first !== nothing
            write_array(group, "first", slot.first)
            write_array(group, "second", slot.second)
        end
    end
    links = create_group(file, "aliases")
    for (i, (name, index)) in enumerate(state.aliases)
        write(links, "name_$i", name)
        write(links, "index_$i", index)
    end
end

function write_checkpoint(f, path, kind)
    destination = abspath(path)
    mktemp(dirname(destination)) do temporary, io
        close(io)
        h5open(temporary, "w") do file
            write(file, "version", 2)
            write(file, "kind", kind)
            f(file)
        end
        Base.Filesystem.rename(temporary, destination)
    end
    return path
end

"""Save model arrays, optimizer state, pending gradients and training randomness."""
function save_checkpoint(path::AbstractString, state::TrainingState)
    write_checkpoint(path, "training") do file
        write_training_state(file, state)
    end
end

function save_checkpoint(path::AbstractString, state::GANState)
    write_checkpoint(path, "GAN") do file
        write_training_state(create_group(file, "generator"), state.generator)
        write_training_state(create_group(file, "discriminator"), state.discriminator)
        write(file, "noise", [state.seed, state.steps])
    end
end

function read_slot_array!(dest, group, key)
    value = read_array(group, key)
    size(dest) == size(value) && eltype(dest) == eltype(value) ||
        throw(DimensionMismatch("checkpoint optimizer array $key differs"))
    return copyto!(dest, value)
end

function read_training_state(file, state)
    state.pending && throw(ArgumentError("complete backward! before loading a checkpoint"))
    seed, forwards = read(file, "random")
    restored = TrainingState(state.model, read_optimizer(file); seed)
    restored.forwards = forwards
    restored.samples, restored.microbatches, restored.epochs = read(file, "progress")
    0 <= restored.microbatches <= restored.samples && restored.epochs >= 0 &&
        (restored.samples == 0) == (restored.microbatches == 0) ||
        throw(ArgumentError("invalid checkpoint progress"))

    links = file["aliases"]
    aliases = [(read(links, "name_$i"), read(links, "index_$i")) for i in 1:read(file, "alias_count")]
    aliases == restored.aliases || throw(ArgumentError("checkpoint parameter names or ties differ"))
    length(restored.slots) == read(file, "parameter_count") || throw(ArgumentError("checkpoint parameter count differs"))
    for (i, slot) in enumerate(restored.slots)
        group = file["parameters/$i"]
        read(group, "name") == restored.names[i] || throw(ArgumentError("checkpoint parameter order differs"))
        slot.lr = read_number(group, "rate")
        validate_rate(slot.lr)
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
    saved = Dict(read(model, "name_$i") => read_array(model, "value_$i") for i in 1:read(file, "array_count"))
    validate_state_arrays(model_arrays(state.model), saved)
    return saved, restored
end

function restore_training_state!(state::TrainingState, saved, restored)
    load_state!(state.model, saved)
    for field in (:run, :optimizer, :slots, :names, :index, :sources, :aliases,
                  :seed, :forwards, :samples, :microbatches, :epochs, :batch, :pending)
        setfield!(state, field, getfield(restored, field))
    end
    return state
end

function read_checkpoint(f, path, kind)
    return h5open(path, "r") do file
        read(file, "version") == 2 || throw(ArgumentError("unsupported checkpoint version"))
        read(file, "kind") == kind || throw(ArgumentError("checkpoint training state differs"))
        f(file)
    end
end

"""Restore a checkpoint into an existing model and training state."""
function load_checkpoint!(state::TrainingState, path::AbstractString)
    payload = read_checkpoint(path, "training") do file
        read_training_state(file, state)
    end
    return restore_training_state!(state, payload...)
end

function load_checkpoint!(state::GANState, path::AbstractString)
    generator, discriminator, noise = read_checkpoint(path, "GAN") do file
        (read_training_state(file["generator"], state.generator),
         read_training_state(file["discriminator"], state.discriminator), read(file, "noise"))
    end
    restore_training_state!(state.generator, generator...)
    restore_training_state!(state.discriminator, discriminator...)
    state.seed, state.steps = noise
    return state
end
