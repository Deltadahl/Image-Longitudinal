using Flux
using Metalhead
using Flux: onehotbatch, onecold, crossentropy, throttle, params, Chain, softmax
using Base.Iterators: repeated
using CUDA
using Statistics
CUDA.math_mode!(CUDA.PEDANTIC_MATH)
include("data_manipulation/data_loader_MNIST.jl")

# Constants
const BATCH_SIZE = 16
const OUTPUT_SIZE_ENCODER = 10
const DEVICE = gpu
const EPOCHS = 2

# Load data
# data_path = "data/data_resized/MNIST_small_224"
data_path = "data/MNIST"
loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE

# Initialize model
# Initialize model
# base_model = EfficientNet(:b4, inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER) |> DEVICE
base_model = ResNet(18; inchannels=1, nclasses=OUTPUT_SIZE_ENCODER) |> DEVICE
model = Chain(base_model, softmax) |> DEVICE

total_params(model) = sum(length, Flux.params(model))
println("Total parameters: ", total_params(model))

# Loss function: cross-entropy loss
loss(x, y) = crossentropy(model(x), y)

# Define an evaluation metric
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# Optimizer: you can use any suitable one, let's take ADAM for instance
optimizer = ADAM()

# Training loop
time_start = time()
for epoch in 1:EPOCHS
    total_accuracy = []
    println("----------------------------------------------")
    println("Epoch: $epoch/$EPOCHS")
    println("----------------------------------------------")
    batch_nr = 0
    while true
        batch_nr += 1
        images, labels = next_batch(loader)

        if images === nothing
            break
        end

        images = images |> DEVICE
        labels = labels |> DEVICE

        # Perform a step of gradient descent
        Flux.train!(loss, params(model), [(images, labels)], optimizer)

        # Print the current accuracy
        acc = accuracy(images, labels)
        push!(total_accuracy, acc)
        if batch_nr % 10 == 0
            range = min(batch_nr-1, 30)
            mean_acc = mean(total_accuracy[end-range:end])
            println("Batch: $batch_nr, Accuracy: $mean_acc")
        end
    end
    loader.idx = 1
    Random.shuffle!(loader.filenames)
    @show (time() - time_start)
end