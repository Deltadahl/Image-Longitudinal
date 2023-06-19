using Flux
# using CUDA
# using Statistics
# using Zygote
using Metalhead
# using Images
include("constants.jl")
# include("data_manipulation/data_loader_MNIST.jl")
# include("data_manipulation/data_loader_OCT.jl")
# n_classes = 10
# model = EffNet("efficientnet-b0"; n_classes=n_classes, in_channels=1)
# model = model |> DEVICE

data_path = "data/MNIST_small"
# data_path = "data/data_resized/all_train_256"
device_test = gpu
# loader = DataLoader(data_path, BATCH_SIZE) |> device_test
# images, labels = next_batch(loader, 1)
# images, labels = next_batch(loader)

images = rand(Float32, 224, 224, 3, 1)

println("size(images) = $(size(images))")

images = images |> device_test

# Not working
# model = EfficientNetv2(:small; inchannels = 1, nclasses = 10)
# model = EfficientNet(:b0; inchannels = 3, nclasses = 100)
# model = EfficientNetv2(:small;)

# Working
# model = ConvNeXt(:tiny;)
# model = ResNet(18; inchannels = 1, nclasses = 10)
# model = VGG(16; pretrain = true)

# Testing
# model = EfficientNet(:b0)
model = EfficientNetv2(:small)

model = model |> device_test
# print summary of the model
for layer in model.layers
    println(layer)
    println()
end
output = model(images)
println(size(output))

total_params(model) = sum(length, Flux.params(model))
println("Total parameters: ", total_params(model))
