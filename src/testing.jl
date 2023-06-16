using Flux
using CUDA
using Statistics
using Zygote
using EfficientNet
using Metalhead
include("constants.jl")
include("data_manipulation/data_loader_MNIST.jl")

# n_classes = 10
# model = EffNet("efficientnet-b0"; n_classes=n_classes, in_channels=1)
# model = model |> DEVICE

data_path = "data/data_resized/MNIST_small_224"
loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE
images, labels = next_batch(loader)
images = images |> DEVICE
labels = labels |> DEVICE

model = EfficientNetv2(:small; inchannels = 1, nclasses = 1000)
model = model |> DEVICE
# print summary of the model
for layer in model.layers
    println(layer)
end
