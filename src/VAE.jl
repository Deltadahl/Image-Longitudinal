# VAE_MNIST.jl
using Flux
using Flux: Chain
using CUDA
using Statistics
using Zygote
using Metalhead
using Colors
CUDA.math_mode!(CUDA.PEDANTIC_MATH)
include("constants.jl")

#Define the encoder
function create_encoder()
    # return Chain(
    #     Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), leakyrelu),
    #     BatchNorm(32),
    #     Dropout(0.2),
    #     Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), leakyrelu),
    #     BatchNorm(64),
    #     Dropout(0.2),
    #     Conv((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
    #     BatchNorm(64),
    #     Dropout(0.2),
    #     Conv((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
    #     BatchNorm(64),
    #     Dropout(0.2),
    #     Flux.flatten,
    # ) |> DEVICE

    model_base = ResNet(18; inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER)
    model = Chain(model_base, relu)


    # model = ResNet(18; pretrain = true)
    # model = ConvNeXt(:tiny;) # Slow and quite bad, but does WORKING
    # model = EfficientNet(:b0, inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER)
    # model = EfficientNetv2(:small, inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER)
    return model
end

# Define the mean and log variance layers
function create_μ_logvar_layers()
    # return Dense(7 * 7 * 64, LATENT_DIM) |> DEVICE,  Dense(7 * 7 * 64, LATENT_DIM) |> DEVICE
    return Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM),  Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM)
end


# Define the decoder
function create_decoder()
    # return Chain(
        #     Dense(LATENT_DIM, 7 * 7 * 64, relu),
        #     x -> reshape(x, (7, 7, 64, :)),
        #     ConvTranspose((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
        #     BatchNorm(64),
        #     Dropout(0.2),
        #     ConvTranspose((3, 3), 64 => 64, stride = 1, pad = SamePad(), leakyrelu),
    #     BatchNorm(64),
    #     Dropout(0.1),
    #     ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), leakyrelu),
    #     BatchNorm(32),
    #     Dropout(0.1),
    #     ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    # ) |> DEVICE

    # return Chain(
        #     Dense(LATENT_DIM, 7 * 7 * 256, relu),
        #     x -> reshape(x, (7, 7, 256, :)),
        #     ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad(), relu),
        #     BatchNorm(128),
        #     # Dropout(0.1),
        #     ConvTranspose((3, 3), 128 => 128, stride = 2, pad = SamePad(), relu),
    #     BatchNorm(128),
    #     # Dropout(0.1),
    #     ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), relu),
    #     BatchNorm(64),
    #     # Dropout(0.1),
    #     ConvTranspose((3, 3), 64 => 64, stride = 2, pad = SamePad(), relu),
    #     BatchNorm(64),
    #     # Dropout(0.1),
    #     ConvTranspose((3, 3), 64 => 1, stride = 2, pad = SamePad(), sigmoid),
    # ) |> DEVICE


    # function my_reshape(x)
    #     return reshape(x, (7, 7, 512, :))
    # end

    return Chain(
        Dense(LATENT_DIM, 7 * 7 * 512, relu),
        x -> reshape(x, (7, 7, 512, :)),
        # my_reshape,
        ConvTranspose((3, 3), 512 => 256, stride = 2, pad = SamePad()),
        BatchNorm(256),
        relu,
        ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad()),
        BatchNorm(128),
        relu,
        ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad()),
        BatchNorm(64),
        relu,
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad()),
        BatchNorm(32),
        relu,
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    )
end

# Define the VAE
mutable struct VAE # TODO test to not have mutable
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

Zygote.@nograd function update_gl_balance!(loss_saver::LossSaver,  kl_div, rec)
    loss_saver.avg_kl += kl_div
    loss_saver.avg_rec += rec
    loss_saver.counter += 1.0f0
end

function convert_to_rgb(images::CuArray{Float32, 4})
    # Broadcast grayscale image along the channel dimension
    return cat(images, images, images, dims=3) |> DEVICE

end

struct RGBReplicationLayer end
Flux.@functor RGBReplicationLayer
(m::RGBReplicationLayer)(x) = cat(x, x, x, dims=3) |> DEVICE

function vgg_subnetworks()
    vgg = VGG(16; pretrain = true)

    # Create subnetworks where second one starts where the first one ends
    vgg_layer2 = Chain(vgg.layers[1][1:2]...) |> DEVICE
    vgg_layer9 = Chain(vgg.layers[1][3:9]...) |> DEVICE # starts from the third layer

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

function normalize(normalizer::LossNormalizer, value::Float32)
    return value / (normalizer.sum / normalizer.count)
end

function vgg_loss(decoded, x, vgg, loss_normalizers)
    normalizing_factors = Dict(
        "loss_mse" => Float32(32.4141831),
        "loss2" => Float32(7.55499029/11.27),
        "loss9" => Float32(0.235879934/4.3),
    )
    weight_factors = Dict(
        "loss_mse" => Float32(1/2),
        "loss2" => Float32(0),
        "loss9" => Float32(1/2),
    )

    (vgg_layer2, vgg_layer9) = vgg
    (loss_normalizer_mse, loss_normalizer2, loss_normalizer9) = loss_normalizers

    # Calculate and normalize the MSE loss
    loss_mse = sum(mean((decoded .- x).^2, dims=(1,2,3)))

    # decoded = convert_to_rgb(decoded)
    # x = convert_to_rgb(x)

    # Use the first subnetwork for the first layer
    decoded = normalize_image_net(decoded)
    x = normalize_image_net(x)
    decoded_feature2 = vgg_layer2(decoded)
    x_feature2 = vgg_layer2(x)

    # # Use the first subnetwork for the first layer
    # decoded_feature2 = vgg_layer2(decoded)
    # x_feature2 = vgg_layer2(x)

    # Calculate and normalize the loss for the first layer
    loss2 = sum(mean((decoded_feature2 .- x_feature2).^2, dims=(1,2,3)))

    # Use the second subnetwork for the second layer, taking the output of the first subnetwork as input
    decoded_feature9 = vgg_layer9(decoded_feature2)
    x_feature9 = vgg_layer9(x_feature2)

    # Calculate and normalize the loss for the second layer
    loss9 = sum(mean((decoded_feature9 .- x_feature9).^2, dims=(1,2,3)))

    loss_mse *= weight_factors["loss_mse"] * normalizing_factors["loss_mse"]
    loss2 *= weight_factors["loss2"] * normalizing_factors["loss2"]
    loss9 *= weight_factors["loss9"] * normalizing_factors["loss9"]
    update_normalizer!(loss_normalizer_mse, loss_mse)
    update_normalizer!(loss_normalizer2, loss2)
    update_normalizer!(loss_normalizer9, loss9)
    if loss_normalizer_mse.count % 10 == 0
        @show loss_normalizer_mse.sum / loss_normalizer_mse.count
        @show loss_normalizer2.sum / loss_normalizer2.count
        @show loss_normalizer9.sum / loss_normalizer9.count
    end
    return loss_mse + loss2 + loss9
end


function loss(m::VAE, x, y, loss_saver::LossSaver, vgg, loss_normalizers, epoch)
    decoded, μ, logvar = m(x)
    # reconstruction_loss = sum(mean((decoded .- x).^2, dims=(1,2,3)))
    reconstruction_loss = vgg_loss(decoded, x, vgg, loss_normalizers)

    kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))
    # kl_divergence = 0.0f0

    # β = 3.2 * 10^(-4) * 4
    # β = 3.2 * 10^(-4) * 10
    β_factor = min(epoch / 10, 5.0f0)

    if β_factor == 4.0f0
        β_factor = 5.0 + cos(epoch / 10 * π)
    end

    β = Float32(10^(-3) * β_factor)

    kl_divergence = β .* kl_divergence
    update_gl_balance!(loss_saver, kl_divergence, reconstruction_loss)
    # if loss_saver.counter % 1000 == 0
    #     @show loss_saver.avg_kl / loss_saver.counter
    #     @show loss_saver.avg_rec / loss_saver.counter
    #     println()
    # end
    return reconstruction_loss + kl_divergence
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
