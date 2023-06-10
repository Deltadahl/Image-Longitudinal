# VAE_OCT.jl
using Flux
using CUDA
using Statistics
using Zygote
using Metalhead
include("constants.jl")

using Flux

function conv_block(ch_in, ch_out)
    layers = Chain(
        Conv((3, 3), ch_in=>ch_out, pad=(1,1), stride=(2,2)),
        BatchNorm(ch_out, relu),
        Conv((3, 3), ch_out=>ch_out, pad=(1,1)),
        BatchNorm(ch_out)
    ) |> DEVICE
    shortcut = Chain(
        Conv((1, 1), ch_in=>ch_out, stride=(2,2)),
        BatchNorm(ch_out)
    ) |> DEVICE
    return (layers, shortcut)
end

function identity_block(ch_in, ch_out)
    layers = Chain(
        Conv((3, 3), ch_in=>ch_out, pad=(1,1)),
        BatchNorm(ch_out, relu),
        Conv((3, 3), ch_out=>ch_out, pad=(1,1)),
        BatchNorm(ch_out)
    ) |> DEVICE
    return (layers, identity)
end

function res_block(x, ch_in, ch_out, project::Bool=false)
    if project
        f, g = conv_block(ch_in, ch_out)
    else
        f, g = identity_block(ch_in, ch_out)
    end
    return Chain(
        x -> (f(x) .+ g(x)),
        relu
    ) |> DEVICE
end

function res_layer(n, ch_in, ch_out)
    res_blocks = [res_block(res_block, ch_in, ch_out, true)]
    append!(res_blocks, [res_block(res_block, ch_out, ch_out) for _ in 1:(n-1)])
    return Chain(res_blocks...) |> DEVICE
end

function ResNet34()
    return Chain(
        Conv((7, 7), 1=>64, pad=(3,3), stride=(2,2)),
        BatchNorm(64, relu),
        MaxPool((3,3), stride=(2,2)),
        res_layer(3, 64, 64),
        res_layer(4, 64, 128),
        res_layer(6, 128, 256),
        res_layer(3, 256, 512),
        AdaptiveMeanPool((1,1)),
        Flux.flatten,
        Dense(512, OUTPUT_RESNET),
        gelu
    ) |> DEVICE
end




function create_encoder()
    # return Chain(
    #     Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), relu) |> DEVICE,
    #     # BatchNorm(32, relu) |> DEVICE,
    #     Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), relu) |> DEVICE,
    #     # BatchNorm(64, relu) |> DEVICE,
    #     Conv((3, 3), 64 => 128, stride = 2, pad = SamePad(), relu) |> DEVICE,
    #     # BatchNorm(128, relu) |> DEVICE,
    #     Conv((3, 3), 128 => 256, stride = 2, pad = SamePad(), relu) |> DEVICE,
    #     # BatchNorm(256, relu) |> DEVICE,
    #     Flux.flatten |> DEVICE,
    #     # Dense(31 * 32 * 256, 1024, relu) |> DEVICE,
    # )
    # return Chain(
    #     Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), leakyrelu),
    #     Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), leakyrelu),
    #     Conv((3, 3), 64 => 128, stride = 2, pad = SamePad(), leakyrelu),
    #     Conv((3, 3), 128 => 256, stride = 2, pad = SamePad(), leakyrelu),
    #     Flux.flatten,
    #     Dense(16 * 16 * 256, ouput_resnet, leakyrelu),
    # ) |> DEVICE


    # block_type = Metalhead.bottleneck
    # block_type = Metalhead.basicblock
    # # block_repeats = [3, 4, 6, 2]
    # block_repeats = [2, 2, 2, 2]

    # return Metalhead.resnet(block_type, block_repeats,
    #     imsize = (512, 512),
    #     inchannels = 1,
    #     nclasses = ouput_resnet) |> DEVICE

    # return ResNet(34; inchannels = 1, nclasses = ouput_resnet) |> DEVICE

    return ResNet34() |> DEVICE
end

function create_μ_logvar_layers()
    return Dense(OUTPUT_RESNET, LATENT_DIM) |> DEVICE, Dense(OUTPUT_RESNET, LATENT_DIM) |> DEVICE
end

function create_decoder()
    # return Chain(
    #     Dense(LATENT_DIM, 16 * 16 * 32, relu),
    #     x -> reshape(x, (16, 16, 32, :)),
    #     ConvTranspose((3, 3), 32 => 256, stride = 1, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    # ) |> DEVICE

    # return Chain(
    #     Dense(LATENT_DIM, 8 * 8 * 32, leakyrelu),
    #     x -> reshape(x, (8, 8, 32, :)),
    #     ConvTranspose((3, 3), 32 => 512, stride = 1, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 512 => 256, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), leakyrelu),
    #     ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    # ) |> DEVICE
    return Chain(
        Dense(LATENT_DIM, 8 * 8 * 32, leakyrelu),
        x -> reshape(x, (8, 8, 32, :)),
        ConvTranspose((3, 3), 32 => 512, stride = 2, pad = SamePad(), leakyrelu),
        ConvTranspose((3, 3), 512 => 256, stride = 2, pad = SamePad(), leakyrelu),
        ConvTranspose((3, 3), 256 => 128, stride = 2, pad = SamePad(), leakyrelu),
        ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), leakyrelu),
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), leakyrelu),
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    ) |> DEVICE
end

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

Zygote.@nograd function log_losses(loss_list_rec, loss_list_kl, reconstruction_loss, kl_divergence)
    push!(loss_list_rec, reconstruction_loss)
    push!(loss_list_kl, kl_divergence)
end

function loss(m::VAE, x, y, loss_list_rec, loss_list_kl)
    decoded, μ, logvar = m(x)
    mse_per_image = mean((decoded - x).^2, dims=(1,2,3))
    reconstruction_loss = sum(mse_per_image)

    # println("mu = $(maximum(μ)), logvar = $(maximum(logvar))")
    kl_divergence = -0.5 .* sum(1 .+ logvar .- μ .^ 2 .- exp.(logvar))
    β = 10^(-5) * 32

    reconstruction_loss = reconstruction_loss / size(x)[4]
    kl_divergence = β * kl_divergence / size(x)[4]
    log_losses(loss_list_rec, loss_list_kl, reconstruction_loss, kl_divergence)

    return reconstruction_loss + kl_divergence
end

Flux.trainable(m::VAE) = (m.encoder, m.μ_layer, m.logvar_layer, m.decoder)
