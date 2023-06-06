# load_model.jl
# using BSON: @load
using Images
using JLD2, FileIO

# include("VAE.jl")
# include("VAE_MNIST.jl")
include("data_manipulation/data_loader.jl")
include("data_manipulation/data_loader_MNIST.jl")
include("constants.jl")
include("VAE_MNIST.jl")

using Glob

function output_image(vae)
    # data_path = "data/data_resized/all_develop"
    data_path = "data/MNIST"
    loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE
    images, labels = next_batch(loader)
    images = images |> DEVICE

    reconstructed, _, _ = vae(images)

    # Convert the reconstructed tensor back to an image
    reconstructed = cpu(reconstructed[:,:,1,1])
    original_image = cpu(images[:,:,1,1])
    reconstructed_image = Images.colorview(Gray, reconstructed)  # remove the singleton dimensions
    original_image = Images.colorview(Gray, original_image)  # remove the singleton dimensions

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
    path_to_original_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-original_image.png")
    save(path_to_image, reconstructed_image)
    save(path_to_original_image, original_image)
    println("Saved image to $path_to_image")

    loader.idx = 1
    Random.shuffle!(loader.filenames)
    return nothing
end


function main()
    # Load the model
    model_path = "saved_models/MNIST_epoch_2_batch_END.jld2"
    vae = load(model_path, "vae")
    vae = vae |> DEVICE

    # encoder = create_encoder()
    # mu_layer, logvar_layer = create_mu_logvar_layers()
    # decoder = create_decoder()
    # vae = VAE(encoder, mu_layer, logvar_layer, decoder) |> DEVICE

    output_image(vae)
    return nothing
end

main()
