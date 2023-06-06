# VAE.jl
using Flux
using CUDA
include("constants.jl")

# Define the encoder
function create_encoder()
    return Chain(
        Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), relu) |> DEVICE,
        Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), relu) |> DEVICE,
        Flux.flatten |> DEVICE,
        Dense(7 * 7 * 64, 128, relu) |> DEVICE,
    )
end

# Define the mean and log variance layers
function create_mu_logvar_layers()
    return Dense(128, 16) |> DEVICE,  Dense(128, 16) |> DEVICE
end

# Define the decoder
function create_decoder()
    return Chain(
        Dense(16, 7 * 7 * 64, relu) |> DEVICE,
        x -> reshape(x, (7, 7, 64, :)),
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), relu) |> DEVICE,
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid) |> DEVICE,
    )
end

# # Define the encoder
# function create_encoder()
#     return Chain(
#         Conv((3, 3), 1 => 32, stride = (1, 1), pad = (1, 1), relu) |> DEVICE,
#         Conv((3, 3), 32 => 64, stride = (2, 2), pad = (1, 1), relu) |> DEVICE,
#         Conv((3, 3), 64 => 64, stride = (2, 2), pad = (1, 1), relu) |> DEVICE,
#         Conv((3, 3), 64 => 64, stride = (1, 1), pad = (1, 1), relu) |> DEVICE,
#         Flux.flatten |> DEVICE,
#     )
# end

# # Define the mean and log variance layers
# function create_mu_logvar_layers()
#     return Dense(3136, 2) |> DEVICE, Dense(3136, 2) |> DEVICE
# end

# # Define the decoder
# function create_decoder()
#     return Chain(
#         Dense(2, 3136, relu) |> DEVICE,
#         x -> reshape(x, (7, 7, 64, :)),
#         ConvTranspose((3, 3), 64 => 64, stride = (1, 1), pad = (1, 1), relu) |> DEVICE,
#         ConvTranspose((3, 3), 64 => 64, stride = (2, 2), pad = (1, 1), relu) |> DEVICE,
#         ConvTranspose((3, 3), 64 => 32, stride = (2, 2), pad = 0, relu) |> DEVICE,
#         ConvTranspose((3, 3), 32 => 1, stride = (1, 1), pad = 0, sigmoid) |> DEVICE,
#         # Trim 29x29 to 28x28
#         x -> x[1:28, 1:28, :, :]
#     )
# end


# function create_encoder()
#     return Chain(
#         Flux.flatten |> DEVICE,
#         Dense(784, 500, relu) |> DEVICE,
#         Dense(500, 300, relu) |> DEVICE,
#     )
# end

# # Define the mean and log variance layers
# function create_mu_logvar_layers()
#     return Dense(300, 20) |> DEVICE, Dense(300, 20) |> DEVICE
# end

# # Define the decoder
# function create_decoder()
#     return Chain(
#         Dense(20, 300, relu) |> DEVICE,
#         Dense(300, 500, relu) |> DEVICE,
#         Dense(500, 784, relu) |> DEVICE,
#         x -> reshape(x, (28, 28, 1, :)),
#     )
# end

# Define the VAE
struct VAE
    encoder::Any
    mu_layer::Any
    logvar_layer::Any
    decoder::Any
end

function reparametrize(mu, logvar)
    r = randn(Float32, size(mu)) |> DEVICE
    return mu + exp.(logvar ./ 2) .* r
end

function (m::VAE)(x)
    encoded = m.encoder(x)
    mu = m.mu_layer(encoded)
    logvar = m.logvar_layer(encoded)
    z = reparametrize(mu, logvar)
    decoded = m.decoder(z)
    return decoded, mu, logvar
end

# Define the loss function
function loss(x, m::VAE)
    decoded, mu, logvar = m(x)
    reconstruction_loss = mse(reshape(decoded, :), reshape(x, :))
    kl_divergence = -0.5 .* sum(1 .+ logvar .- mu .^ 2 .- exp.(logvar))
    l = reconstruction_loss + kl_divergence
    return l
end
