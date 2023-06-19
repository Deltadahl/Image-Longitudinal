using Flux
using Metalhead
using Flux: onehotbatch, onecold, crossentropy, throttle, params, Chain, softmax
using Base.Iterators: repeated
using CUDA
using Statistics
using DataLoaders
using Random
using Images
using FileIO
CUDA.math_mode!(CUDA.PEDANTIC_MATH)

# Constants
const BATCH_SIZE = 16
const OUTPUT_SIZE_ENCODER = 10
const DEVICE = gpu
const EPOCHS = 2

# Load data
# data_path = "data/data_resized/MNIST_small_224"
data_path = "data/MNIST"
filenames = readdir(data_path)
Random.shuffle!(filenames)

function get_label(filename::String)
    for i in 0:9
        if startswith(filename, string(i))
            return i
        end
    end
    error("Invalid filename: ", filename)
end

# Load all data into memory
images = []
labels = []
for filename in filenames
    image = load(joinpath(data_path, filename))
    # Convert the image to grayscale and then to Float32
    image = Float32.(Gray.(image))
    image = imresize(image, (224, 224))

    # Reshape the image to the format (height, width, channels, batch size)
    image = reshape(image, size(image)..., 1, 1)
    push!(images, image)
    push!(labels, get_label(filename))
end

# Concatenate all images along the 4th dimension to form a single batch
images = cat(images..., dims=4) |> DEVICE
labels = Flux.onehotbatch(labels, 0:9) |> DEVICE

# Create DataLoader
data = (images, labels)
dataloader = DataLoader(data, BATCH_SIZE)

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
    for (images, labels) in dataloader
        batch_nr += 1

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
    Random.shuffle!(filenames)
    @show (time() - time_start)
end
