# generate_image.jl
using Images
using JLD2, FileIO
using Glob
using Flux
using CUDA
include("VAE.jl")

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

function output_image(vae, loader; epoch=0)
    images, labels = first(loader)
    images = images |> DEVICE

    encoded = vae.encoder(images)
    μ = vae.μ_layer(encoded)
    reconstructed = vae.decoder(μ)

    reconstructed_x2 = vae.decoder(vae.μ_layer(vae.encoder(reconstructed)))

    # Convert the reconstructed tensor back to an image
    reconstructed = cpu(reconstructed[:,:,1,1])
    original_image = cpu(images[:,:,1,1])
    reconstructed_x2 = cpu(reconstructed_x2[:,:,1,1])

    reconstructed_image = Images.colorview(Gray, reconstructed)
    original_image = Images.colorview(Gray, original_image)
    reconstructed_x2 = Images.colorview(Gray, reconstructed_x2)


    # Save the reconstructed image
    if !isdir(OUTPUT_IMAGE_DIR)  # make output directory, (git ignored)
        mkdir(OUTPUT_IMAGE_DIR)
    end

    # Get the latest integer used in the saved files
    png_files = glob("*-reconstructed_image.png", OUTPUT_IMAGE_DIR)
    if isempty(png_files)
        new_integer = 1
    else
        latest_integer = maximum(parse(Int, match(r"^(\d+)-", basename(file)).captures[1]) for file in png_files)
        new_integer = latest_integer + 1
    end

    # Update the path_to_image to use the new integer
    path_to_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-$epoch-reconstructed_image.png")
    path_to_original_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-$epoch-original_image.png")
    save(path_to_image, reconstructed_image)
    save(path_to_original_image, original_image)
    println("Saved to $new_integer")

    generated_image = generate_image(vae)
    path_to_image = joinpath(OUTPUT_IMAGE_DIR, "$new_integer-$epoch-generated_image.png")
    save(path_to_image, generated_image)

    return nothing
end
