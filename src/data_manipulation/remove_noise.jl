using Distributed

# Add the number of cores to use
addprocs(8)

@everywhere begin
    using BM3DDenoise
    using ImageIO
    using FileIO
    using Images
    using Glob
    using Printf

    # Function to denoise the image
    function denoise_image(img::AbstractArray, noise_variance::Float64)::AbstractArray
        img_array = float(channelview(img))  # Convert image to 2D array
        img_denoised = bm3d(img_array, noise_variance)  # Apply BM3D algorithm
        img_denoised = (img_denoised .- minimum(img_denoised)) ./ (maximum(img_denoised) - minimum(img_denoised)) # Normalize
        img_denoised = colorview(Gray, img_denoised)  # Convert back to grayscale image
        return img_denoised
    end

    # Function to process each file
    function process_file(file::String, noise_variance::Float64, base_path::String, base_path_modified::String)
        # Maintain the subfolder structure in the new path
        relative_path = relpath(file, base_path)
        save_path = joinpath(base_path_modified, relative_path)

        # Check if the file already exists in the output directory. If it does, then continue with the next iteration.
        if isfile(save_path)
            println("File already exists -> Skipping.")
            return
        end

        img = load(file)
        img_denoised = denoise_image(img, noise_variance)

        mkpath(dirname(save_path))
        save(save_path, img_denoised)
    end
end

function main()
    base_path = "data/CellData/OCT_white_to_black"
    base_path_modified = "data/CellData/OCT_mb3d"
    noise_variance = 0.15
    subfolders = [
        # "test/NORMAL",
        # "test/DRUSEN",
        # "test/DME",
        # "test/CNV",
        "train/DRUSEN",
        "train/NORMAL",
        "train/CNV",
        # "train/DME",
    ]

    println("Starting the main task")
    # Flatten the file structure
    all_files = [glob("*.jpeg", joinpath(base_path, subfolder)) for subfolder in subfolders]
    all_files = vcat(all_files...)  # Concatenate all the files into a single array

    # @sync @distributed for file in all_files
    #     process_file(file, noise_variance, base_path, base_path_modified)
    #     println("Finished processing $file")
    # end

    start_time = time()
    for (i, file) in enumerate(all_files)
        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

        process_file(file, noise_variance, base_path, base_path_modified)

        println("Finished processing $file")
    end
end

main()
