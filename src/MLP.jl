# MLP.jl
using CUDA
# Define a simple feedforward neural network
struct SimpleNN
    layer1::Any
    layer2::Any
    layer3::Any
    output::Any
end

function create_simple_nn()
    return SimpleNN(
        Dense(253952, 1024, relu) |> DEVICE,  # Layer 1
        Dense(1024, 512, relu) |> DEVICE,    # Layer 2
        Dense(512, 1024, relu) |> DEVICE,    # Layer 3
        Dense(1024, 253952, sigmoid) |> DEVICE  # Output layer
    )
end

function (m::SimpleNN)(x)
    x = Flux.flatten(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.output(x)
    return reshape(x, (496, 512, 1, :)), 0, 0
end

function loss(x, m::SimpleNN)
    decoded, _, _ = m(x)
    # decoded_flat = Flux.flatten(decoded)
    # x_flat = Flux.flatten(x)
    reconstruction_loss = mse(reshape(decoded, :), reshape(x, :))
    return reconstruction_loss
end
