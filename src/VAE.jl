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
    return Chain(
        Dense(LATENT_DIM, 7 * 7 * 512, relu),
        x -> reshape(x, (7, 7, 512, :)),
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

# Zygote.@nograd function log_losses(loss_list_rec, loss_list_kl, reconstruction_loss, kl_divergence)
#     push!(loss_list_rec, reconstruction_loss)
#     push!(loss_list_kl, kl_divergence)
# end

# function loss(m::VAE, x, y, loss_list_rec, loss_list_kl)
#     decoded, μ, logvar = m(x)
#     mse_per_image = mean((decoded - x).^2, dims=(1,2,3))
#     reconstruction_loss = sum(mse_per_image)
#     kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))
#     β = 10^(-5) * 32

#     reconstruction_loss = reconstruction_loss / size(x)[4]
#     kl_divergence = β * kl_divergence / size(x)[4]
#     log_losses(loss_list_rec, loss_list_kl, reconstruction_loss, kl_divergence)

#     return reconstruction_loss + kl_divergence
# end

# mutable struct KLBalance
#     avg_rec_loss::Float32
#     avg_kl::Float32
# end

# Zygote.@nograd function update_kl_balance!(kl_balance::KLBalance, rec_loss, kl_div)
#     kl_balance.avg_rec_loss += rec_loss
#     kl_balance.avg_kl += kl_div
# end

# function loss(m::VAE, x, y, kl_balance::KLBalance)
#     decoded, μ, logvar = m(x)
#     reconstruction_loss = sum(mean((decoded - x).^2, dims=(1,2,3)))
#     kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))

#     β = 3.2 * 10^(-4) * 4
#     reconstruction_loss = reconstruction_loss / size(x)[4]
#     kl_divergence = β * kl_divergence / size(x)[4]

#     update_kl_balance!(kl_balance, reconstruction_loss, kl_divergence)
#     return reconstruction_loss + kl_divergence
# end

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

# function convert_to_rgb(images::CuArray{Float32, 4})
#     H, W, C, B = size(images)

#     # Create an empty array with 3 color channels
#     rgb_images = Array{Float32, 4}(undef, 224, 224, 3, B)

#     for b in 1:B
#         resized_image = imresize(images[:, :, 1, b], (224, 224))
#         for c in 1:3
#             rgb_images[:, :, c, b] = resized_image
#         end
#     end
#     return rgb_images
# end

# function convert_to_rgb(images::CuArray{Float32, 4})
#     H, W, C, B = size(images)

#     # Create an empty array with 3 color channels
#     rgb_images = CuArray{Float32, 4}(undef, 224, 224, 3, B)

#     for b in 1:B
#         # TODO check if copy
#         resized_image = images[:,:,1,b] #CuArrays.cu(imresize(Array(images[:, :, 1, b]), (224, 224)))
#         for c in 1:3
#             rgb_images[:, :, c, b] = resized_image
#         end
#     end
#     return rgb_images
# end

# function convert_to_rgb(images::CuArray{Float32, 4})
#     # Resize the last dimension to 3
#     size_images = size(images)
#     # Preallocate a CuArray of zeros with an extra 3rd dimension for RGB channels
#     rgb_images = Flux.zeros(eltype(images), size_images[1], size_images[2], 3, size_images[4]) |> DEVICE

#     # Fill each channel with the grayscale image
#     for channel in 1:3
#         rgb_images[:, :, channel, :] = images
#     end
#     return rgb_images
# end

# function convert_to_rgb(images::CuArray{Float32, 4})
#     # Repeat grayscale image along the channel dimension
#     rgb_images = repeat(images, outer=[1,1,3,1])
#     return rgb_images |> DEVICE
# end

function convert_to_rgb(images::CuArray{Float32, 4})
    # Broadcast grayscale image along the channel dimension
    rgb_images = cat(images, images, images, dims=3)
    return rgb_images |> DEVICE
end


# function vgg_loss(decoded, x, vgg)
#     decoded = convert_to_rgb(decoded)
#     x = convert_to_rgb(x)
#     decoded = vgg(decoded)
#     x = vgg(x)
#     return sum(mean((decoded .- x).^2, dims=(1,2,3))) # TODO need to change when I change VGG16 output.
# end

# function vgg_loss(decoded, x, vgg)
#     decoded = convert_to_rgb(decoded)
#     x = convert_to_rgb(x)
#     decoded_features = vgg(decoded)
#     x_features = vgg(x)
#     loss = 0.0f0
#     weights = Dict("layer_2" => 0.7f0, "layer_9" => 0.3f0)  # assign weights to layers
#     for layer in keys(decoded_features)
#         loss += weights[layer] * sum(mean((decoded_features[layer] .- x_features[layer]).^2, dims=(1,2,3)))
#     end
#     return loss
# end

function vgg_subnetworks()
    vgg = VGG(16; pretrain = true)

    # Create subnetworks where second one starts where the first one ends
    vgg_layer2 = Chain(vgg.layers[1][1:2]...) |> DEVICE
    vgg_layer9 = Chain(vgg.layers[1][3:9]...) |> DEVICE # starts from the third layer

    return (vgg_layer2, vgg_layer9)
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

function normalize(normalizer::LossNormalizer, value::Float32)
    return value / (normalizer.sum / normalizer.count)
end

# function vgg_loss(decoded, x, vgg)
#     (vgg_layer2, vgg_layer9) = vgg

#     loss_mse = sum(mean((decoded .- x).^2, dims=(1,2,3)))

#     decoded = convert_to_rgb(decoded)
#     x = convert_to_rgb(x)

#     # Use the first subnetwork for the first layer
#     decoded_feature2 = vgg_layer2(decoded)
#     x_feature2 = vgg_layer2(x)

#     # Calculate the loss for the first layer
#     loss2 = sum(mean((decoded_feature2 .- x_feature2).^2, dims=(1,2,3)))

#     # Use the second subnetwork for the second layer, taking the output of the first subnetwork as input
#     decoded_feature9 = vgg_layer9(decoded_feature2)
#     x_feature9 = vgg_layer9(x_feature2)

#     # Calculate the loss for the second layer
#     loss9 = sum(mean((decoded_feature9 .- x_feature9).^2, dims=(1,2,3)))

#     return loss2 * 0.6f0 + loss9 * 0.2f0 + loss_mse * 0.2f0
# end
function vgg_loss(decoded, x, vgg, loss_normalizers)
    (vgg_layer2, vgg_layer9) = vgg
    (loss_normalizer_mse, loss_normalizer2, loss_normalizer9) = loss_normalizers

    # Calculate and normalize the MSE loss
    loss_mse = sum(mean((decoded .- x).^2, dims=(1,2,3)))
    update_normalizer!(loss_normalizer_mse, loss_mse)
    loss_mse = normalize(loss_normalizer_mse, loss_mse)

    decoded = convert_to_rgb(decoded)
    x = convert_to_rgb(x)

    # Use the first subnetwork for the first layer
    decoded_feature2 = vgg_layer2(decoded)
    x_feature2 = vgg_layer2(x)

    # Calculate and normalize the loss for the first layer
    loss2 = sum(mean((decoded_feature2 .- x_feature2).^2, dims=(1,2,3)))
    update_normalizer!(loss_normalizer2, loss2)
    loss2 = normalize(loss_normalizer2, loss2)

    # Use the second subnetwork for the second layer, taking the output of the first subnetwork as input
    decoded_feature9 = vgg_layer9(decoded_feature2)
    x_feature9 = vgg_layer9(x_feature2)

    # Calculate and normalize the loss for the second layer
    loss9 = sum(mean((decoded_feature9 .- x_feature9).^2, dims=(1,2,3)))
    update_normalizer!(loss_normalizer9, loss9)
    loss9 = normalize(loss_normalizer9, loss9)

    # Weights for the losses might need to be adjusted based on your needs
    return loss_mse * 0.1f0 + loss2 * 0.65f0 + loss9 * 0.25f0
end


function loss(m::VAE, x, y, loss_saver::LossSaver, vgg, loss_normalizers)
    decoded, μ, logvar = m(x)
    reconstruction_loss = sum(mean((decoded .- x).^2, dims=(1,2,3)))
    # reconstruction_loss = vgg_loss(decoded, x, vgg, loss_normalizers)

    kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))

    # β = 3.2 * 10^(-4) * 4
    β = 3.2 * 10^(-4) * 10

    kl_divergence = β .* kl_divergence

    update_gl_balance!(loss_saver, kl_divergence, reconstruction_loss)
    return reconstruction_loss + kl_divergence
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
