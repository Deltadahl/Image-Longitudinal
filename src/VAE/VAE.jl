# VAE.jl
using Flux
using Flux: Chain
using CUDA
using Statistics
using Zygote
using Metalhead
using Colors
using NNlib
CUDA.math_mode!(CUDA.PEDANTIC_MATH)
include("constants.jl")

# Define the encoder
function create_encoder()
    model_base = ResNet(18; inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER)
    model = Chain(model_base, relu)
    return model
end

# Define the mean and log variance layers
function create_μ_logvar_layers()
    return Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM),  Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM)
end

# Define a function for the reshape operation
function my_reshape(x)
    return reshape(x, (7, 7, 1024, :))
end

function create_decoder()
    return Chain(
        Dense(LATENT_DIM, 7 * 7 * 1024, relu),
        my_reshape,
        Conv((3,3), 1024 => 512, pad = SamePad()),
        BatchNorm(512),
        relu,
        Upsample(2),  # upsampling
        Conv((3,3), 512 => 512, pad = SamePad()),
        BatchNorm(512),
        relu,
        Conv((3,3), 512 => 256, pad = SamePad()),
        BatchNorm(256),
        relu,
        Upsample(2),  # upsampling
        Conv((3,3), 256 => 256, pad = SamePad()),
        BatchNorm(256),
        relu,
        Conv((3,3), 256 => 128, pad = SamePad()),
        BatchNorm(128),
        relu,
        Upsample(2),  # upsampling
        Conv((3,3), 128 => 128, pad = SamePad()),
        BatchNorm(128),
        relu,
        Conv((3,3), 128 => 64, pad = SamePad()),
        BatchNorm(64),
        relu,
        Upsample(2),  # upsampling
        Conv((3,3), 64 => 64, pad = SamePad()),
        BatchNorm(64),
        relu,
        Conv((3,3), 64 => 32, pad = SamePad()),
        BatchNorm(32),
        relu,
        Upsample(2),  # upsampling
        Conv((3,3), 32 => 32, pad = SamePad()),
        BatchNorm(32),
        relu,
        Conv((3,3), 32 => 1, pad = SamePad()),
        sigmoid
    )
end

# Define the VAE
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

mutable struct LossSaver
    avg_kl::Float32
    avg_rec::Float32
    counter::Float32
end

Zygote.@nograd function update_kl_rec!(loss_saver::LossSaver,  kl_div, rec)
    loss_saver.avg_kl += kl_div
    loss_saver.avg_rec += rec
    loss_saver.counter += 1.0f0
end

# From gray to RGB helper
struct RGBReplicationLayer end
Flux.@functor RGBReplicationLayer
(m::RGBReplicationLayer)(x) = cat(x, x, x, dims=3) |> DEVICE

function vgg_subnetworks()
    vgg = VGG(16; pretrain = true)

    # Create subnetworks where second one starts where the first one ends
    vgg_layer2 = Chain(vgg.layers[1][1:2]...) |> DEVICE
    vgg_layer9 = Chain(vgg.layers[1][3:9]...) |> DEVICE # starts from the third layer

    # Make the input 3 channels instead of 1
    vgg_layer2_gray = Chain(RGBReplicationLayer(), vgg_layer2)

    return (vgg_layer2_gray, vgg_layer9)
end

mutable struct LossNormalizer
    sum::Float32
    count::Int32
    LossNormalizer() = new(0.0f0, 0)
end

Zygote.@nograd function update_normalizer!(normalizer::LossNormalizer, value::Float32)
    normalizer.sum += value
    normalizer.count += 1
    return nothing
end

function normalize_image_net(images::CuArray{Float32, 4})
    mean = 0.449
    std_dev = 0.226
    normalized_images = (images .- mean) ./ std_dev

    return normalized_images
end

function vgg_loss(decoded, x, vgg, loss_normalizers)
    normalizing_factors = Dict(
        "loss_mse" => Float32(400),
        "loss2" => Float32(6.25),
        "loss9" => Float32(1),
    )

    weight_factors = Dict(
        "loss_mse" => Float32(1/3),
        "loss2" => Float32(1/3),
        "loss9" => Float32(1/3),
    )

    (vgg_layer2, vgg_layer9) = vgg
    (loss_normalizer_mse, loss_normalizer2, loss_normalizer9) = loss_normalizers

    loss_mse = mean(mean((decoded .- x).^2, dims=(1,2,3)))

    decoded = normalize_image_net(decoded)
    x = normalize_image_net(x)
    decoded_feature2 = vgg_layer2(decoded)
    x_feature2 = vgg_layer2(x)

    loss2 = mean(mean((decoded_feature2 .- x_feature2).^2, dims=(1,2,3)))

    # Use the second subnetwork, taking the output of the first subnetwork as input
    decoded_feature9 = vgg_layer9(decoded_feature2)
    x_feature9 = vgg_layer9(x_feature2)

    loss9 = mean(mean((decoded_feature9 .- x_feature9).^2, dims=(1,2,3)))

    loss_mse *= weight_factors["loss_mse"] * normalizing_factors["loss_mse"]
    loss2 *= weight_factors["loss2"] * normalizing_factors["loss2"]
    loss9 *= weight_factors["loss9"] * normalizing_factors["loss9"]
    update_normalizer!(loss_normalizer_mse, loss_mse)
    update_normalizer!(loss_normalizer2, loss2)
    update_normalizer!(loss_normalizer9, loss9)
    return loss_mse + loss2 + loss9
end

mutable struct StatisticsSaver
    mu_mean_values::Vector{Float32}
    mu_variance_values::Vector{Float32}
    logvar_mean_values::Vector{Float32}
    logvar_variance_values::Vector{Float32}
end

StatisticsSaver() = StatisticsSaver(Float32[], Float32[], Float32[], Float32[])

Zygote.@nograd function update_statistics!(statistics_saver::StatisticsSaver, μ, logvar)
    # Calculate the mean and variance of μ for the batch and append to the arrays
    push!(statistics_saver.mu_mean_values, mean(μ))
    push!(statistics_saver.mu_variance_values, var(μ))

    # Calculate the mean and variance of logvar for the batch and append to the arrays
    push!(statistics_saver.logvar_mean_values, mean(logvar))
    push!(statistics_saver.logvar_variance_values, var(logvar))
end

function loss(m::VAE, x, loss_saver::LossSaver, vgg, loss_normalizers, β_nr, statistics_saver::StatisticsSaver, training::Bool, epoch)
    decoded, μ, logvar = m(x)
    reconstruction_loss = vgg_loss(decoded, x, vgg, loss_normalizers)
    kl_divergence = mean(-0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar), dims=1))

    β_max = 0.037558578f0
    # Linearly increase β from 0 to β_max in the first 5 epochs
    β_factor = min(β_max * β_nr / 5, β_max)
    β = Float32(β_factor)

    kl_divergence = β .* kl_divergence
    update_kl_rec!(loss_saver, kl_divergence, reconstruction_loss)
    if training
        update_statistics!(statistics_saver, μ, logvar)
    end
    return reconstruction_loss + kl_divergence
end

# Train all components of the VAE.
Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
