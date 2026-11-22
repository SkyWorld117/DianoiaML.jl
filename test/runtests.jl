using DianoiaML
using LinearAlgebra: I, dot
using Random: MersenneTwister, seed!
using Test

function dense_model(inputs, outputs, activation=None)
    model = Sequential()
    model.add_layer(model, Dense; input_shape=(inputs,), layer_size=outputs,
                    activation_function=activation, randomization=false)
    return model
end

function finite_difference!(objective, array, index; h=1f-3)
    original = array[index]
    array[index] = original + h
    plus = objective()
    array[index] = original - h
    minus = objective()
    array[index] = original
    return (plus - minus) / (2h)
end

@testset "DianoiaML" begin
    include("gradients.jl")
    include("training.jl")
    include("models.jl")
    include("composition.jl")
    include("training_state.jl")
end
