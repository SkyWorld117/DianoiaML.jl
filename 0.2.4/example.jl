using MLDatasets, BenchmarkTools
include("./framework.jl")
using .Yisy_AI_Framework

train_x, train_y = MNIST.traindata()
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

@btime begin
    model = Sequential()
    model.add_layer(model, Convolutional_2D(input_filter=1, filter=2, input_size=784, input2D_size=(28,28), kernel_size=(3,3), activation_function=ReLU))
    model.add_layer(model, Convolutional_2D(input_filter=2, filter=4, input_size=def, input2D_size=(26,26), kernel_size=(3,3), activation_function=ReLU))
    model.add_layer(model, MaxPooling_2D(input_filter=4, input_size=def, input2D_size=(24,24), kernel_size=(2,2), activation_function=None))
    model.add_layer(model, Dense(input_size=def, layer_size=128, activation_function=ReLU))
    model.add_layer(model, Dense(input_size=def, layer_size=10, activation_function=Softmax))

    Adam.fit(sequential=model, input_data=flatten(train_x, 3), output_data=One_Hot(train_y, 10, dict), loss_function=Cross_Entropy_Loss, monitor=Absolute_Loss, α=0.02, epochs=5, mini_batch=128)
end
