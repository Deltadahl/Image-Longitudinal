# VAE.jl
using Flux
using CUDA
include("constants.jl")

#Define the encoder
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

# function create_encoder()
#     return Chain(
#         Flux.flatten |> DEVICE,
#         Dense(784, 784, relu) |> DEVICE,
#     )
# end

# # Define the mean and log variance layers
# function create_mu_logvar_layers()
#     return Dense(784, 784) |> DEVICE, Dense(784, 784) |> DEVICE
# end

# # Define the decoder
# function create_decoder()
#     return Chain(
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
    return mu #+ exp.(logvar ./ 2) .* r # TODO remove comment
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
    # reconstruction_loss = mse(reshape(decoded, :), reshape(x, :))
    reconstruction_loss = Flux.Losses.mse(decoded, x)
    kl_divergence = -0.5 .* sum(1 .+ logvar .- mu .^ 2 .- exp.(logvar))
    l = reconstruction_loss + 0.0 * kl_divergence # TODO remove Weight on kl
    return l
end
