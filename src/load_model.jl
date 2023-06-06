# load_model.jl
# using BSON: @load
using Images
using JLD2, FileIO

include("VAE.jl")
include("data_manipulation/data_loader.jl")

using Glob

function output_image(vae)
    loader = DataLoader("data/data_resized/all_develop", BATCH_SIZE) |> DEVICE
    images, labels = next_batch(loader)
    images = images |> DEVICE

    reconstructed, _, _ = vae(images)

    # Convert the reconstructed tensor back to an image
    reconstructed = cpu(reconstructed[:,:,:,1])
    reconstructed_image = Images.colorview(Gray, dropdims(reconstructed, dims=3))  # remove the singleton dimensions

    # Save the reconstructed image
    if !isdir(OUTPUT_IMAGE_DIR)  # make output directory, (git ignored)
        mkdir(OUTPUT_IMAGE_DIR)
    end

    # Get the latest integer used in the saved files
    png_files = glob("*-reconstructed_image.png", OUTPUT_IMAGE_DIR)
    if isempty(png_files)
        new_integer = 1
    else
        latest_integer = maximum(parse(Int, match(r"(\d+)-reconstructed_image.png", file).captures[1]) for file in png_files)
        new_integer = latest_integer + 1
    end

    # Update the path_to_image to use the new integer
    path_to_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-reconstructed_image.png")
    save(path_to_image, reconstructed_image)
    println("Saved image to $path_to_image")

    loader.idx = 1
    Random.shuffle!(loader.filenames)
    return nothing
end


function main()
    # Load the model
    # @load "saved_models/vae.bson" vae
    vae = load("saved_models/epoch_1_batch_END_vae.jld2", "vae")
    vae = vae |> DEVICE
    output_image(vae)
    return nothing
end

main()
