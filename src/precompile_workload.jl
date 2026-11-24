using PrecompileTools: @compile_workload

@compile_workload begin
    for T in (Float32, Float16, BFloat16, Float64, BigFloat)
        model = Sequential(
            Conv2D(1, 2; kernel_size=(3, 3), activation=ReLU(), precision=T),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(18, 2; activation=ReLU(), precision=T),
            Dense(2, 3; precision=T);
            input_shape=(8, 8, 1),
        )
        inputs = fill(T(1//2), 8, 8, 1, 4)
        targets = zeros(T, 3, 4)
        targets[1, :] .= one(T)
        state = TrainingState(model, Adam())
        fit!(state, batches(inputs, targets; batch_size=4, shuffle=false);
             loss=CrossEntropyWithLogits(), verbose=false)
    end
end
