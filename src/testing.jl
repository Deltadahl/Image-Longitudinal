using Flux
using CUDA
using Statistics
using Zygote
using Metalhead
using Images
include("constants.jl")
include("data_manipulation/data_loader_MNIST.jl")
# include("data_manipulation/data_loader_OCT.jl")
# n_classes = 10
# model = EffNet("efficientnet-b0"; n_classes=n_classes, in_channels=1)
# model = model |> DEVICE

data_path = "data/MNIST_small"
# data_path = "data/data_resized/all_train_256"
device_test = gpu
loader = DataLoader(data_path, BATCH_SIZE) |> device_test
# images, labels = next_batch(loader, 1)
images, labels = next_batch(loader)

# TODO resize images to 224x224
println("size(images) = $(size(images))")
# images = imresize.(images, (224, 224))

function prepare_image(img)
    # `img` should be a 2D grayscale image
    # Convert to RGB
    img_rgb = repeat(img, outer=(1, 1, 3))

    # Resize to 224x224
    img_resized = Images.imresize(img_rgb, (224, 224))

    return img_resized
end

# images_resized = [imresize(images[:,:,1,i], (224, 224)) for i in 1:size(images, 4)]
# images_resized = [prepare_image(images[:,:,:,i]) for i in 1:size(images, 4)]
# Convert back to 4D tensor
# images = cat(images_resized..., dims=4)

println("size(images) = $(size(images))")

images = images |> device_test
labels = labels |> device_test

# model = EfficientNetv2(:small; inchannels = 1, nclasses = 10)
# model = EfficientNet(:b0; inchannels = 3, nclasses = 100)

# model = EfficientNetv2(:small;)

model = ConvNeXt(:tiny;) # WORKING
# model = ResNet(18; inchannels = 1, nclasses = 10)

# model = VGG(16; pretrain = true)

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
