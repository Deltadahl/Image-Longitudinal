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
    # data_path = "data/MNIST"
    data_name = "OCT"
    data_path = "data/data_resized/bm3d_224_train"
    save_nr = 105

    model_path = "saved_models/save_nr_$(save_nr).jld2"
    vae = load(model_path, "vae")

    vae.encoder = vae.encoder |> DEVICE
    vae.μ_layer = vae.μ_layer |> DEVICE
    vae.logvar_layer = vae.logvar_layer |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE

    if data_name == "OCT"
        loader = DataLoaderOCT(data_path, BATCH_SIZE, false)
    else
        loader = DataLoaderMNIST(data_path, BATCH_SIZE)
    end
    output_image(vae, loader)
    println("Done!")
    return nothing
end

main()
