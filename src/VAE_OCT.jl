# VAE_OCT.jl
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
function create_μ_logvar_layers()
    return Dense(1024, 512) |> DEVICE, Dense(1024, 512) |> DEVICE
end

# Define the decoder
function create_decoder()
    return Chain(
        Dense(512, 31 * 32 * 256, relu) |> DEVICE,
        x -> reshape(x, (31, 32, 256, :)),
        ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(128, relu) |> DEVICE,
        ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(64, relu) |> DEVICE,
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), relu) |> DEVICE,
        BatchNorm(32, relu) |> DEVICE,
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid) |> DEVICE,
    )
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
    return μ + exp.(logvar ./ 2) .* r
end

function (m::VAE)(x)
    encoded = m.encoder(x)
    μ = m.μ_layer(encoded)
    logvar = m.logvar_layer(encoded)
    z = reparametrize(μ, logvar)
    decoded = m.decoder(z)
    return decoded, μ, logvar
end

# Define the loss function
function loss(m::VAE, x, y)
    decoded, μ, logvar = m(x)
    reconstruction_loss = Flux.Losses.mse(decoded, x)
    kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))
    l = reconstruction_loss + kl_divergence
    return l
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
