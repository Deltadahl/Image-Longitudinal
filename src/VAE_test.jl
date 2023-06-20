# VAE_MNIST_test.jl
using Flux
using Flux: Chain, MLUtils
using Statistics
include("constants.jl")

function create_encoder()
    return Chain(
        Flux.flatten,
        Dense(28*28, OUTPUT_SIZE_ENCODER, relu),
    )
end

function create_μ_logvar_layers()
    return Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM),  Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM)
end

function create_decoder()
    return Chain(
        Dense(LATENT_DIM, 7 * 7 * 64, relu),
        x -> reshape(x, (7, 7, 64, :)),
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad()),
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    )
end

mutable struct VAE
    encoder::Chain
    μ_layer::Dense
    logvar_layer::Dense
    decoder::Chain
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


function loss(m::VAE, x)
    decoded, μ, logvar = m(x)
    reconstruction_loss = sum(mean((decoded .- x).^2, dims=(1,2,3)))
    kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))
    β = 3.2 * 10^(-4) * 10
    kl_divergence = β .* kl_divergence
    return reconstruction_loss + kl_divergence
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
