using BenchmarkTools
include("./framework.jl")
using .Yisy_AI_Framework

data = [0 1 2 3; 4 5 6 7; 8 9 0 1; 2 3 4 5]
data = convert(Array{Float32}, data)

data2 = zeros(Float32, (2,5))

let
    model = Sequential()
    model.add_layer(model, Dense(input_size=4, layer_size=5, randomization=true, activation_function=ReLU))
    model.add_layer(model, Dense(input_size=5, layer_size=5, randomization=true, activation_function=tanH))
    model.add_layer(model, Dense(input_size=5, layer_size=5, randomization=true, activation_function=ReLU))
    model.add_layer(model, Dense(input_size=5, layer_size=2, randomization=true, activation_function=Sigmoid))

    Stochastic_Gradient_Descent.fit(sequential=model, input_data=data, output_data=data2, loss_function=Quadratic_Loss, learning_rate=0.02, epochs=10, mini_batch=32)

    save_sequential(model, "./model0.h5")
end

model = load_sequential("./model0.h5")
println(model.layers[2].input_size)
