using PrecompileTools: @compile_workload

# Exercise the layer types and loss used by the README's MNIST model.
@compile_workload begin
    for T in (Float32, Float16, BFloat16, Float64, BigFloat)
        model = Sequential(; precision=T)
        add_layer!(model, Conv2D; input_shape=(8, 8, 1), filter=2,
                   kernel_size=(3, 3), activation_function=ReLU, randomization=false)
        add_layer!(model, MaxPooling2D; pool_size=(2, 2))
        add_layer!(model, Flatten)
        add_layer!(model, Dense; layer_size=2, activation_function=ReLU, randomization=false)
        add_layer!(model, Dense; layer_size=3, activation_function=Softmax, randomization=false)
        inputs = fill(T(1//2), 8, 8, 1, 4)
        targets = zeros(T, 3, 4)
        targets[1, :] .= one(T)
        Adam.fit(; model, input_data=inputs, output_data=targets,
                 loss_function=Categorical_Cross_Entropy_Loss, monitor=Classification,
                 epochs=1, batch=4, shuffle=false, verbose=false)
    end
end
