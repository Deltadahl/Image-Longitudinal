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

#Define the encoder
function create_encoder()
    model_base = ResNet(18; inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER)
    model = Chain(model_base, relu)
    return model
end

# Define the mean and log variance layers
function create_μ_logvar_layers()
    return Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM),  Dense(OUTPUT_SIZE_ENCODER, LATENT_DIM)
end

# Define a named function for the reshape operation
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


# # Define the decoder
# function create_decoder()
#     return Chain(
#         Dense(LATENT_DIM, 7 * 7 * 1024, relu),
#         my_reshape,
#         ConvTranspose((3, 3), 1024 => 512, stride = 1, pad = SamePad()),
#         BatchNorm(512),
#         relu,
#         Conv((3,3), 512 => 512, pad = SamePad()),
#         BatchNorm(512),
#         relu,
#         ConvTranspose((3, 3), 512 => 256, stride = 2, pad = SamePad()),
#         BatchNorm(256),
#         relu,
#         Conv((3,3), 256 => 256, pad = SamePad()),
#         BatchNorm(256),
#         relu,
#         ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad()),
#         BatchNorm(128),
#         relu,
#         Conv((3,3), 128 => 128, pad = SamePad()),
#         BatchNorm(128),
#         relu,
#         ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad()),
#         BatchNorm(64),
#         relu,
#         Conv((3,3), 64 => 64, pad = SamePad()),
#         BatchNorm(64),
#         relu,
#         ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad()),
#         BatchNorm(32),
#         relu,
#         Conv((3,3), 32 => 32, pad = SamePad()),
#         BatchNorm(32),
#         relu,
#         ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
#     )
# end

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

struct RGBReplicationLayer end
Flux.@functor RGBReplicationLayer
(m::RGBReplicationLayer)(x) = cat(x, x, x, dims=3) |> DEVICE

function vgg_subnetworks()
    vgg = VGG(16; pretrain = true)

    # Create subnetworks where second one starts where the first one ends
    vgg_layer2 = Chain(vgg.layers[1][1:2]...) |> DEVICE
    vgg_layer9 = Chain(vgg.layers[1][3:9]...) |> DEVICE # starts from the third layer

    vgg_layer2_gray = Chain(RGBReplicationLayer(), vgg_layer2)

    # # Function to print details of Conv layers
    # function print_conv_details(layer)
    #     println("Conv Layer: input channels = $(size(layer.weight, 3)), output channels = $(size(layer.weight, 4)), kernel size = $(size(layer.weight, 1))x$(size(layer.weight, 2))")
    # end

    # # Print out the layers of the full VGG16 network
    # println("VGG16 layers:")
    # for (i, layer) in enumerate(vgg.layers)
    #     println("Layer $i: $(typeof(layer))")
    #     if layer isa Conv
    #         print_conv_details(layer)
    #     end
    # end

    # # Print out the layers of the first subnetwork
    # println("vgg_layer2_gray layers:")
    # for (i, layer) in enumerate(vgg_layer2_gray.layers)
    #     println("Layer $i: $(typeof(layer))")
    #     if layer isa Conv
    #         print_conv_details(layer)
    #     end
    # end

    # # Print out the layers of the second subnetwork
    # println("vgg_layer9 layers:")
    # for (i, layer) in enumerate(vgg_layer9.layers)
    #     println("Layer $i: $(typeof(layer))")
    #     if layer isa Conv
    #         print_conv_details(layer)
    #     end
    # end

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
        # "loss_mse" => Float32(20),
        "loss_mse" => Float32(400),
        "loss2" => Float32(6.25),
        "loss9" => Float32(1),
    )
    # normalizing_factors = Dict(
    #     "loss_mse" => Float32(400),
    #     "loss2" => Float32(400/64),
    #     "loss9" => Float32(1),
    # )

    weight_factors = Dict(
        # "loss_mse" => Float32(1/3),
        # "loss2" => Float32(1/3),
        # "loss9" => Float32(1/3),
        "loss_mse" => Float32(1/3),
        "loss2" => Float32(1/3),
        "loss9" => Float32(1/3),
    )

    (vgg_layer2, vgg_layer9) = vgg
    (loss_normalizer_mse, loss_normalizer2, loss_normalizer9) = loss_normalizers

    loss_mse = mean(mean((decoded .- x).^2, dims=(1,2,3)))

    # Use the first subnetwork for the first layer
    decoded = normalize_image_net(decoded)
    x = normalize_image_net(x)
    decoded_feature2 = vgg_layer2(decoded)
    x_feature2 = vgg_layer2(x)
    # Divide loss2 by the number of channels, the width and the height of the feature map
    loss2 = mean(mean((decoded_feature2 .- x_feature2).^2, dims=(1,2,3)))

    # Use the second subnetwork for the second layer, taking the output of the first subnetwork as input
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
    sum_mu::Float32
    sum_logvar::Float32
    sum_mu2::Float32
    sum_logvar2::Float32
    counter::Float32
end

StatisticsSaver() = StatisticsSaver(0f0, 0f0, 0f0, 0f0, 0f0)

Zygote.@nograd function update_statistics!(statistics_saver::StatisticsSaver, μ, logvar)
    statistics_saver.sum_mu += sum(μ)
    statistics_saver.sum_logvar += sum(logvar)
    statistics_saver.sum_mu2 += sum(μ .^ 2)
    statistics_saver.sum_logvar2 += sum(exp.(logvar))  # Convert logvar to var
    statistics_saver.counter += size(μ, 1) * size(μ, 2)  # BATCH_SIZE
end



function loss(m::VAE, x, loss_saver::LossSaver, vgg, loss_normalizers, β_nr, statistics_saver::StatisticsSaver, training::Bool)
    decoded, μ, logvar = m(x)
    reconstruction_loss = vgg_loss(decoded, x, vgg, loss_normalizers)
    # reconstruction_loss = sum(mean((decoded .- x).^2, dims=(1,2,3))) * Float32(24.4704336/0.8072882)
    kl_divergence = mean(-0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar), dims=1))


    β_max = 7.0f0 * 0.3
    β_factor = min(β_max * β_nr, β_max)
    # β_factor = β_max * β_nr # TODO REMVOE THIS LINE AFTER TEST! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    β = Float32(β_factor / 111.82532)

    kl_divergence = β .* kl_divergence
    update_kl_rec!(loss_saver, kl_divergence, reconstruction_loss)
    if training
        update_statistics!(statistics_saver, μ, logvar)
    end
    return reconstruction_loss + kl_divergence
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)


########

# function vgg_loss(decoded, x, vgg, loss_normalizers)
#     normalizing_factors = Dict(
#         "loss_mse" => Float32(1),
#         "loss2" => Float32(1),
#         "loss9" => Float32(1/25),
#     )

#     weight_factors = Dict(
#         "loss_mse" => Float32(1/3),
#         "loss2" => Float32(1/3),
#         "loss9" => Float32(1/3),
#     )

#     (vgg_layer2, vgg_layer9) = vgg
#     (loss_normalizer_mse, loss_normalizer2, loss_normalizer9) = loss_normalizers

#     loss_mse = sum(mean((decoded .- x).^2, dims=(1,2,3))) / (224 * 224 * 1)

#     # Use the first subnetwork for the first layer
#     decoded = normalize_image_net(decoded)
#     x = normalize_image_net(x)
#     decoded_feature2 = vgg_layer2(decoded)
#     x_feature2 = vgg_layer2(x)
#     # Divide loss2 by the number of channels, the width and the height of the feature map
#     loss2 = sum(mean((decoded_feature2 .- x_feature2).^2, dims=(1,2,3))) / (224 * 224 * 64)

#     # Use the second subnetwork for the second layer, taking the output of the first subnetwork as input
#     decoded_feature9 = vgg_layer9(decoded_feature2)
#     x_feature9 = vgg_layer9(x_feature2)

#     loss9 = sum(mean((decoded_feature9 .- x_feature9).^2, dims=(1,2,3))) / (56 * 56 * 256)

#     loss_mse *= weight_factors["loss_mse"] * normalizing_factors["loss_mse"]
#     loss2 *= weight_factors["loss2"] * normalizing_factors["loss2"]
#     loss9 *= weight_factors["loss9"] * normalizing_factors["loss9"]
#     update_normalizer!(loss_normalizer_mse, loss_mse)
#     update_normalizer!(loss_normalizer2, loss2)
#     update_normalizer!(loss_normalizer9, loss9)
#     return loss_mse + loss2 + loss9
# end

# function loss(m::VAE, x, loss_saver::LossSaver, vgg, loss_normalizers, β_nr)
#     decoded, μ, logvar = m(x)
#     reconstruction_loss = vgg_loss(decoded, x, vgg, loss_normalizers)
#     # reconstruction_loss = sum(mean((decoded .- x).^2, dims=(1,2,3))) * Float32(24.4704336/0.8072882)
#     kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar)) / (224 * 224 * 64 * 0.698908248)

#     # β_max = 7.0 * 0.8
#     β_max = 7.0f0
#     β_factor = min(β_max * β_nr, β_max)
#     β = Float32(10^(-3) * β_factor)

#     kl_divergence = β .* kl_divergence
#     update_kl_rec!(loss_saver, kl_divergence, reconstruction_loss)
#     return reconstruction_loss + kl_divergence
# end

# Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
