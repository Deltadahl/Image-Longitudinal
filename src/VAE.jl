# VAE.jl
using Flux
using CUDA
include("constants.jl")

# Define the encoder
function create_encoder()
    return Chain(
        Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(32, relu) |> DEVICE,
        Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(64, relu) |> DEVICE,
        Conv((3, 3), 64 => 128, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(128, relu) |> DEVICE,
        Conv((3, 3), 128 => 256, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(256, relu) |> DEVICE,
        Flux.flatten |> DEVICE,
        Dense(31 * 32 * 256, 1024, relu) |> DEVICE,
    )
end

# Define the mean and log variance layers
function create_mu_logvar_layers()
    return Dense(1024, 512) |> DEVICE, Dense(1024, 512) |> DEVICE
end

# Define the decoder
function create_decoder()
    return Chain(
        Dense(512, 31 * 32 * 256, relu),
        x -> reshape(x, (31, 32, 256)),
        ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad(), relu),
        BatchNorm(128, relu),
        ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), relu),
        BatchNorm(64, relu),
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), relu),
        BatchNorm(32, relu),
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    ) |> DEVICE
end

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
