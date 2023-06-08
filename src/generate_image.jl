# load_model.jl
using Images
using JLD2, FileIO
using Glob

# include("VAE.jl")
# include("data_manipulation/data_loader.jl")
include("VAE_MNIST.jl")
include("data_manipulation/data_loader_MNIST.jl")
include("constants.jl")

function generate_image(vae::VAE)
    # Sample a point in the latent space
    z = randn(Float32, LATENT_DIM) |> DEVICE

    # Use the decoder to generate an image
    generated = vae.decoder(z)

    # Reshape the generated tensor and convert it to an image
    generated_image = cpu(generated[:,:,1,1])
    generated_image = Images.colorview(Gray, generated_image)

    return generated_image
end

function output_image(vae)
    # data_path = "data/data_resized/all_develop"
    data_path = "data/MNIST_small"
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

    generated_image = generate_image(vae)
    path_to_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-generated_image.png")
    save(path_to_image, generated_image)
    println("Saved image to $path_to_image")

    return nothing
end


function main()
    # Load the model
    model_path = "saved_models/MNIST_epoch_20_batch_END.jld2"
    vae = load(model_path, "vae")
    vae = vae |> DEVICE

    output_image(vae)
    return nothing
end

main()
