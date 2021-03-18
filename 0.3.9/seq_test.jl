using MLDatasets, LinearAlgebra
BLAS.set_num_threads(1)
include("./YisyAIFramework.jl")
using .YisyAIFramework

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)


begin
    model = Sequential()
    model.add_layer(model, Conv2D; filter=16, input_size=784, input2D_size=(28,28), kernel_size=(3,3), activation_function=ReLU)
    model.add_layer(model, Conv2D; filter=32, kernel_size=(3,3), activation_function=ReLU)
    model.add_layer(model, MaxPooling2D; kernel_size=(2,2), activation_function=None)
    model.add_layer(model, Dense; layer_size=128, activation_function=ReLU)
    model.add_layer(model, Dense; layer_size=10, activation_function=Softmax_CEL)

    Adam.fit(model=model, input_data=flatten(train_x, 3), output_data=One_Hot(train_y, 10, dict),
            loss_function=Categorical_Cross_Entropy_Loss, monitor=Classification, epochs=10, batch=128)
end
