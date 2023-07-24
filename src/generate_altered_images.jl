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

    # selected_features = [75, 25, 98, 18, 107, 101, 33, 60, 52, 34, 47, 116, 102, 5, 113, 43, 128, 14, 41, 16, 38, 12, 4, 115, 105, 26, 64, 32, 1, 103, 59, 63, 21, 56, 31, 72, 46, 36, 108, 104, 89, 82, 124, 97, 122, 53, 84, 55, 57, 65, 6, 94, 39, 15, 78, 74, 49, 50, 96, 7, 83, 81, 93, 17, 20, 69, 87, 125, 112, 28, 90, 110, 40, 117, 2, 99, 10, 88, 118, 29, 30, 42, 23, 119, 67, 19, 44, 126, 70, 73, 45, 114, 35, 106, 86, 109, 85, 77, 71, 62, 54, 51, 13, 91, 123, 66, 27, 58, 61, 68, 11, 37, 76, 48, 120, 8, 3, 80, 79, 121, 100, 92, 24, 9, 111, 95, 22, 127]
    # selected_features = [50, 19, 94, 10, 77, 70, 47, 84, 9, 8, 81, 29, 79, 41, 4, 63, 62, 7, 12, 14, 99, 51, 45, 91, 37, 64, 5, 33, 123, 121, 116, 112, 44, 127, 6, 105, 111, 126, 72, 114, 60, 122, 53, 54, 16, 24, 125, 26, 95, 49, 115, 119, 35, 93, 109, 11, 40, 102, 2, 39, 78, 68, 120, 103, 46, 100, 75, 17, 20, 42, 23, 25, 71, 104, 18, 97, 55, 56, 107, 3, 80, 38, 128, 30, 69, 67, 22, 108, 74, 65, 117, 110, 66, 43, 13, 61, 101, 32, 87, 98, 76, 52, 1, 92, 21, 86, 28, 96, 113, 34, 36, 57, 58, 59, 15, 31, 89, 83, 118, 124, 90, 73, 82, 85, 88, 106, 27, 48]
    # selected_features = [53, 7, 25, 1, 87, 61, 45]
    selected_features = [92, 111, 50, 3, 91, 67, 37, 8, 90, 120, 54, 56, 21, 61, 75, 29, 80, 12, 95, 118, 73, 94, 101, 20, 48, 99, 104, 13, 59, 52, 106, 79, 4, 86, 93, 85, 72, 32, 87, 35, 47, 113, 40, 53, 36, 55, 122, 22, 5, 2, 88, 77, 26, 15, 7, 108, 58, 28, 39, 128, 126, 25, 103, 65, 105, 34, 18, 69, 27, 43, 64, 123, 38, 78, 17, 121, 42, 49, 33, 66, 57, 6, 24, 112, 10, 115, 68, 45, 11, 51, 41, 97, 70, 102, 114, 89, 71, 44, 110, 109, 62, 31, 124, 16, 1, 74, 9, 119, 14, 83, 117, 76, 60, 46, 23, 84, 98, 82, 100, 107, 81, 125, 127, 30, 19, 96, 63, 116]
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
        # 1: - up, + down, sometimes disease in the middle
        # 2: - up, + down, sometimes disease in the middle
        # 3: - incline, + decline
        # 4: - lighter image, + darker image
        # 5: - cancauve, + convex
        # 6: - right larger and brighter, + left larger and brighter



        # 1: ? blurr \in [-1, 1]
        # 2: ? blurr \in [-1.5, 0]
        # 3: - inclined, + declined
        # 4: - lighter image, + darker image
        # 5: - convex, + concave
        # 6: - white on left (upper) side, + NOT white on left (upper) side
        # 7: - left larger, + right larger

        # 1: - down, + up (white on left side for abs(X) > 2)
        # 2: - up, + down, blurr \in [-0.6, 1.6]
        # 3: - decline, + incline
        # 4: - darker, + lighter
        # 5: - convex, + concave

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
    save_nr = 269

    model_path = "saved_models/save_nr_$(save_nr).jld2"
    vae = load(model_path, "vae")
    device = cpu
    output_image(vae, device)
end

main()
