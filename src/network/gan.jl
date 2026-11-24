"""Keep generator and discriminator as independently composable models."""
struct GAN{G<:AbstractLayer,D<:AbstractLayer}
    generator::G
    discriminator::D
    function GAN(generator::G, discriminator::D) where {G<:AbstractLayer,D<:AbstractLayer}
        shape(discriminator, shape(generator))
        new{G,D}(generator, discriminator)
    end
end

layer_children(model::GAN) = (; generator=model.generator, discriminator=model.discriminator)
generate(model::GAN, noise) = predict(model.generator, noise)

mutable struct GANState{G,D,F}
    generator::G
    discriminator::D
    noise::F
    seed::UInt64
    steps::UInt64
end

"""
    GANState(model; noise, generator_optimizer=Adam(), discriminator_optimizer=Adam(), seed)

The noise function receives a random stream and a batch size.
"""
function GANState(model::GAN; noise, generator_optimizer=Adam(), discriminator_optimizer=Adam(),
                  seed::Integer=rand(default_rng(), UInt64))
    generator = TrainingState(model.generator, generator_optimizer; seed)
    discriminator = TrainingState(model.discriminator, discriminator_optimizer; seed=UInt64(seed) ⊻ 1)
    any(value -> haskey(discriminator.index, value), keys(generator.index)) &&
        throw(ArgumentError("generator and discriminator must own distinct parameters"))
    return GANState(generator, discriminator, noise, UInt64(seed), UInt64(0))
end

function gan_target(prediction::AbstractMatrix, real)
    outputs = size(prediction, 1)
    outputs in (1, 2) || throw(DimensionMismatch("GAN discriminator needs one or two outputs"))
    target = zero(prediction)
    if real
        target[end, :] .= one(eltype(target))
    elseif outputs == 2
        target[1, :] .= one(eltype(target))
    end
    return target
end

function discriminator_step!(state, input, real, loss, monitor)
    prediction = forward!(state, input)
    target = gan_target(prediction, real)
    score = monitor === nothing ? 0 : monitor(prediction, target)
    backward!(state, target; loss)
    step!(state)
    return score
end

"""Fit a GAN on batches of real examples."""
function fit!(state::GANState, data; loss::AbstractLoss=BinaryCrossEntropyWithLogits(),
              monitor=nothing, epochs::Int=1, verbose::Bool=true)
    epochs >= 0 || throw(ArgumentError("epochs must be nonnegative"))
    generator, discriminator = state.generator, state.discriminator
    for _ in 1:epochs
        seen, score = 0, 0.0
        for real in (applicable(data) ? data() : data)
            count = batch_examples(real)
            rng = Xoshiro((UInt128(state.seed) << 64) | UInt128(state.steps))
            noise = state.noise(rng, count)
            batch_examples(noise) == count || throw(DimensionMismatch("GAN noise batch size differs"))
            score += discriminator_step!(discriminator, real, true, loss, monitor)
            generated = forward!(generator, noise)
            score += discriminator_step!(discriminator, generated, false, loss, monitor)
            prediction = forward!(discriminator, generated)
            derivative = backpropagate!(discriminator, gan_target(prediction, true), loss; accumulate=false)
            backward!(generator, derivative)
            step!(generator)
            seen += count
            state.steps += 1
        end
        seen > 0 || throw(ArgumentError("the batch iterator produced no examples"))
        generator.epochs += 1
        discriminator.epochs += 1
        verbose && monitor !== nothing && println("Epoch ", generator.epochs, " mean monitor ", score / seen)
    end
    return state
end
