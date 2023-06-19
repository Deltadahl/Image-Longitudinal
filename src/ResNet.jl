using Flux
include("constants.jl")

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
        Dense(512, OUTPUT_SIZE_ENCODER),
        relu
    ) |> DEVICE
end
