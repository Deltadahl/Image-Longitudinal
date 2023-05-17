using Flux
using Flux: @epochs, mse, params
using Flux: Conv, Dense, MaxPool, BatchNorm, flatten
using Flux: Chain, @functor

# Define the encoder
function create_encoder()
    return Chain(
        Conv((3, 3), 1=>32, stride=2, pad=SamePad(), relu),
        Conv((3, 3), 32=>64, stride=2, pad=SamePad(), relu),
        Conv((3, 3), 64=>128, stride=2, pad=SamePad(), relu),
        Conv((3, 3), 64=>256, stride=2, pad=SamePad(), relu),
        flatten,
        Dense(31 * 32 * 256, 1024, relu) # TODO might just ignore this layer?
    )
end

# Define the mean and log variance layers
function create_mu_logvar_layers()
    return Dense(1024, 512), Dense(1024, 512)
end

# Define the decoder
function create_decoder()
    return Chain(
        Dense(512, 31 * 32 * 256, relu),
        x -> reshape(x, (31, 32, 256, :)),
        ConvTranspose((3, 3), 256=>128, stride=2, pad=SamePad(), relu),
        ConvTranspose((3, 3), 128=>64, stride=2, pad=SamePad(), relu),
        ConvTranspose((3, 3), 64=>32, stride=2, pad=SamePad(), relu),
        ConvTranspose((3, 3), 32=>1, stride=2, pad=SamePad(), sigmoid)
    )
end

# Define the VAE
struct VAE
    encoder
    mu_layer
    logvar_layer
    decoder
end

@functor VAE

function (m::VAE)(x)
    encoded = m.encoder(x)
    mu = m.mu_layer(encoded)
    logvar = m.logvar_layer(encoded)
    # z = mu + exp.(0.5 .* logvar) #.* CUDA.randn(Float32, size(mu)) # TODO uncomment
    r = randn(Float32, size(mu)) |> gpu # TODO see if there is a better workaround and double check that no differentiation is happening
    z = mu + exp.(0.5 .* logvar) .* r
    decoded = m.decoder(z)
    return decoded, mu, logvar
end

# Define the loss function
function loss(x, m::VAE)
    decoded, mu, logvar = m(x)
    reconstruction_loss = mse(decoded, x)
    kl_divergence = -0.5 .* sum(1 .+ logvar .- mu.^2 .- exp.(logvar))
    return reconstruction_loss + kl_divergence
end
