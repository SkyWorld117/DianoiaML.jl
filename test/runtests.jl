using DianoiaML
using LinearAlgebra: I, dot
using Random: MersenneTwister, seed!
using Test

function configured_dense(weights::Matrix{T}, activation=Identity()) where {T<:AbstractFloat}
    layer = Dense(size(weights, 2), size(weights, 1); activation, precision=T, randomization=false)
    layer.weights .= weights
    return layer
end

dense_model(input, output; precision=Float32) = Sequential(Dense(input, output; precision, randomization=false))

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
    @test_throws ArgumentError gpu_backend(:cuda)
    include("gradients.jl")
    include("training.jl")
    include("models.jl")
    include("composition.jl")
    include("training_state.jl")
    include("execution_plan.jl")
    include("precision.jl")
    include("transformer.jl")
end
