include("YisyAIFramework.jl")
using .YisyAIFramework

function noise()
    return reshape([rand(1.0f0:1.0f0:10.0f0), rand(Float32)], (2,1))
end

model = GAN(noise)

model.add_Glayer(model, Dense; input_shape=(2,), layer_size=16, activation_function=ReLU)
model.add_Glayer(model, Dense; layer_size=64, activation_function=ReLU)
model.add_Glayer(model, Constructive; shape=(8,8,1))
model.add_Glayer(model, UpSampling2D; size=(2,2), activation_function=None)
model.add_Glayer(model, Flatten;)
model.add_Glayer(model, Dense; layer_size=256, activation_function=ReLU)
model.add_Glayer(model, Dense; layer_size=784, activation_function=ReLU)
model.add_Glayer(model, Constructive; shape=(28,28,1))

model.add_Dlayer(model, Conv2D; filter=16, kernel_size=(3,3), activation_function=ReLU)
model.add_Dlayer(model, Conv2D; filter=32, kernel_size=(3,3), activation_function=ReLU)
model.add_Dlayer(model, MaxPooling2D; pool_size=(2,2))
model.add_Dlayer(model, Flatten;)
model.add_Dlayer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_Dlayer(model, Dense; layer_size=64, activation_function=ReLU)
model.add_Dlayer(model, Dense; layer_size=2, activation_function=Sigmoid)

model.initialize(model, 1)

save_GAN(model, "model0.h5")
model1 = load_GAN("model0.h5", noise)