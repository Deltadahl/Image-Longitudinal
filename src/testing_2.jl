using Flux
using Metalhead
using CUDA

CUDA.math_mode!(CUDA.PEDANTIC_MATH)
device_test = gpu
images = randn(Float32, 224, 224, 3, 1) |> device_test
println("size(images) = $(size(images))")
# model = EfficientNetv2(:small)
model = ResNet(18; inchannels=3, nclasses=1000)
# model = VGG(16; pretrain = true)
model = model |> device_test

for layer in model.layers
    println(layer)
    println()
end
output = model(images)
println(size(output))

total_params(model) = sum(length, Flux.params(model))
println("Total parameters: ", total_params(model))


