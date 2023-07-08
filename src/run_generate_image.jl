# generate_image.jl
using Images
using JLD2, FileIO
using Glob
using Flux
using CUDA
include("generate_image.jl")
include("data_manipulation/data_loader_MNIST.jl")
include("data_manipulation/data_loader_OCT.jl")

function main()
    # data_name = "MNIST"
    # data_path = "data/MNIST_small"
    data_name = "OCT"
    data_path = "data/data_resized/bm3d_224_train" # have train here just to see what the images look like
    epoch = 66

    model_path = "saved_models/$(data_name)_epoch_$(epoch).jld2"
    vae = load(model_path, "vae")

    vae.encoder = vae.encoder |> DEVICE
    vae.μ_layer = vae.μ_layer |> DEVICE
    vae.logvar_layer = vae.logvar_layer |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE

    if data_name == "OCT"
        loader = DataLoaderOCT(data_path, BATCH_SIZE, false) # Have true here just to see what the images look like
    else
        loader = DataLoaderMNIST(data_path, BATCH_SIZE)
    end
    output_image(vae, loader)
    return nothing
end

main()
