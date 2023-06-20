# load_model.jl
using Images
using JLD2, FileIO
using Glob
using Flux
using CUDA

# include("VAE.jl")
# include("data_manipulation/data_loader.jl")
include("VAE.jl")
include("data_manipulation/data_loader_MNIST.jl")
include("data_manipulation/data_loader_OCT.jl")
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

function output_image(vae, loader)
    images, labels = first(loader)
    @show size(images)
    # show type of images
    @show typeof(images)
    images = images |> DEVICE

    # reconstructed, _, _ = vae(images)
    encoded = vae.encoder(images)
    μ = vae.μ_layer(encoded)
    @show size(μ)
    reconstructed = vae.decoder(μ)

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

    generated_image = generate_image(vae)
    path_to_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-generated_image.png")
    save(path_to_image, generated_image)
    println("Saved image to $path_to_image")

    # ----

    # encoded = vae.encoder(images)
    # μ = vae.μ_layer(encoded)
    # println("size mu = $(size(μ))")
    # μ[3,:] .+= μ[3,:] .+ 3.0
    # decoded = vae.decoder(μ)
    # # Convert the reconstructed tensor back to an image
    # reconstructed = cpu(decoded[:,:,1,1])
    # reconstructed_image = Images.colorview(Gray, reconstructed)  # remove the singleton dimensions
    # path_to_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-reconstructed_image_altered.png")
    # save(path_to_image, reconstructed_image)

    return nothing
end


function main()
    data_name = "MNIST"
    data_path = "data/MNIST_small"
    # data_name = "OCT"
    # data_path = "data/data_resized/bm3d_496_512_test"
    epoch = 2

    model_path = "saved_models/$(data_name)_epoch_$(epoch).jld2"
    vae = load(model_path, "vae")
    vae = vae |> DEVICE

    if data_name == "OCT"
        loader = DataLoaderOCT(data_path, BATCH_SIZE, false)
    else
        loader = DataLoaderMNIST(data_path, BATCH_SIZE) |> DEVICE
    end
    output_image(vae, loader)
    return nothing
end

main()
