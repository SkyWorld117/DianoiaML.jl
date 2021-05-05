using MLDatasets
include("./YisyAIFramework.jl")
using .YisyAIFramework

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)


model = Sequential()
model.add_layer(model, Conv2D; filter=16, input_shape=(28,28,1), kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, Conv2D; filter=32, kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Flatten;)
model.add_layer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_layer(model, Dense; layer_size=10, activation_function=Softmax_CEL)

Adam.fit(model=model, input_data=Array{Float32}(reshape(train_x, 28,28,1,60000)), output_data=oneHot(train_y, 10, dict),
        loss_function=Categorical_Cross_Entropy_Loss, monitor=Classification, epochs=10, batch=128)

#=for i in 1:5
    SGD.fit(model=model, input_data=Array{Float32}(reshape(train_x, 28,28,1,60000)), output_data=One_Hot(train_y, 10, dict),
            loss_function=Categorical_Cross_Entropy_Loss, monitor=Classification, epochs=10, batch=128)
end=#

#=
model.initialize(model, 1)
train_x = Array{Float32}(train_x)
for i in 1:1000
    data = Array{Float32}(reshape(train_x[:,:,rand(1:60000)], 28,28,1,1))
    println(size(data))
    model.activate(model, data)
    println("activate")
end
println("Done")
=#