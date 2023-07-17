# generate_image.jl
using Images
using JLD2, FileIO
using Glob
using Flux
using CUDA
include("VAE.jl")

function output_image(vae, device)

    vae.encoder = vae.encoder |> device
    vae.μ_layer = vae.μ_layer |> device
    vae.logvar_layer = vae.logvar_layer |> device
    vae.decoder = vae.decoder |> device
    vae = vae |> device

    selected_features = [75, 25, 98, 18, 107, 101, 33, 60, 52, 34, 47, 116, 102, 5, 113, 43, 128, 14, 41, 16, 38, 12, 4, 115, 105, 26, 64, 32, 1, 103, 59, 63, 21, 56, 31, 72, 46, 36, 108, 104, 89, 82, 124, 97, 122, 53, 84, 55, 57, 65, 6, 94, 39, 15, 78, 74, 49, 50, 96, 7, 83, 81, 93, 17, 20, 69, 87, 125, 112, 28, 90, 110, 40, 117, 2, 99, 10, 88, 118, 29, 30, 42, 23, 119, 67, 19, 44, 126, 70, 73, 45, 114, 35, 106, 86, 109, 85, 77, 71, 62, 54, 51, 13, 91, 123, 66, 27, 58, 61, 68, 11, 37, 76, 48, 120, 8, 3, 80, 79, 121, 100, 92, 24, 9, 111, 95, 22, 127]
    output_altered_dir = "output_altered_images"

    # Save the reconstructed image
    if !isdir(output_altered_dir)  # make output directory, (git ignored)
        mkdir(output_altered_dir)
    end

    # Get the latest integer used in the saved directories
    directories = readdir(output_altered_dir; join=true)
    filtered_dirs = filter(isdir, directories)

    if isempty(filtered_dirs)
        new_integer = 1
    else
        latest_integer = maximum(parse(Int, match(r"^(\d+)", basename(dir)).captures[1]) for dir in filtered_dirs)
        new_integer = latest_integer + 1
    end

    output_altered_dir = joinpath(output_altered_dir, "$new_integer")


    # Update the path_to_image to use the new integer
    # Sample a point in the latent space
    z = randn(Float32, LATENT_DIM) |> device
    range_max = 4
    for i in -range_max:0.4:range_max
    # for i in 128:-1:1
        nr = round(range_max + i, digits=1)
        path_to_image = joinpath(output_altered_dir, "$nr.png")
        z[selected_features[7]] = i
        # 1: ? blurr \in [-1, 1]
        # 2: ? blurr \in [-1.5, 0]
        # 3: - inclined, + declined
        # 4: - lighter image, + darker image
        # 5: - convex, + concave
        # 6: - white on left (upper) side, + NOT white on left (upper) side
        # 7: - left larger, + right larger

        # z[selected_features[i]] = -2


        # Use the decoder to generate an image
        generated = vae.decoder(z)
        # Reshape the generated tensor and convert it to an image
        generated_image = cpu(generated[:,:,1,1])
        generated_image = Images.colorview(Gray, generated_image)

        save(path_to_image, generated_image)
        println("Done with $i")
    end

    return nothing
end

function main()
    save_nr = 626

    model_path = "saved_models/save_nr_$(save_nr).jld2"
    vae = load(model_path, "vae")
    device = cpu
    output_image(vae, device)
end

main()



