# VAE_MNIST.jl
using Flux
using CUDA
using Statistics
include("constants.jl")

#Define the encoder
function create_encoder()
    return Chain(
        Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), leakyrelu),
        # BatchNorm(32),
        Dropout(0.1),
        Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), leakyrelu),
        # BatchNorm(64),
        Dropout(0.1),
        # Conv((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
        # BatchNorm(64),
        # Dropout(0.2),
        # Conv((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
        # BatchNorm(64),
        # Dropout(0.2),
        Flux.flatten,
    ) |> DEVICE
end

# Define the mean and log variance layers
function create_μ_logvar_layers()
    return Dense(7 * 7 * 64, LATENT_DIM) |> DEVICE,  Dense(7 * 7 * 64, LATENT_DIM) |> DEVICE
end

# Define the decoder
function create_decoder()
    return Chain(
        Dense(LATENT_DIM, 7 * 7 * 64, relu),
        x -> reshape(x, (7, 7, 64, :)),
        # ConvTranspose((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
        # BatchNorm(64),
        # Dropout(0.2),
        ConvTranspose((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
        # BatchNorm(64),
        Dropout(0.1),
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), leakyrelu),
        # BatchNorm(32),
        Dropout(0.1),
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    ) |> DEVICE
end

# Define the VAE
struct VAE
    encoder::Any
    μ_layer::Any
    logvar_layer::Any
    decoder::Any
end

function reparametrize(μ, logvar)
    r = randn(Float32, size(μ)) |> DEVICE
    return μ .+ exp.(logvar ./ 2) .* r
end

function (m::VAE)(x)
    encoded = m.encoder(x)
    μ = m.μ_layer(encoded)
    logvar = m.logvar_layer(encoded)
    z = reparametrize(μ, logvar)
    decoded = m.decoder(z)
    return decoded, μ, logvar
end

function loss(m::VAE, x, y)
    decoded, μ, logvar = m(x)
    # reconstruction_loss = Flux.Losses.mse(decoded, x) * size(x)[4]
    mse_per_image = mean((decoded - x).^2, dims=(1,2,3))
    reconstruction_loss = sum(mse_per_image)
    kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))
    β = 10^(-5) * 32
    return (reconstruction_loss + β * kl_divergence) / size(x)[4]
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
