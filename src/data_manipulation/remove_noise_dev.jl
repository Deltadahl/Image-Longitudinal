using Distributed
using BM3DDenoise
using ImageIO
using FileIO
using Images
using Glob

# Add the number of cores to use
addprocs(16)

@everywhere begin
    # Function to denoise the image
    function denoise_image(img::AbstractArray, noise_variance::Float64)::AbstractArray
        img_array = float(channelview(img))  # Convert image to 2D array
        img_denoised = bm3d(img_array, noise_variance)  # Apply BM3D algorithm
        img_denoised = colorview(Gray, img_denoised)  # Convert back to grayscale image
        return img_denoised
    end

    # Function to process each file
    function process_file(file::String, noise_variance::Float64, base_path::String, base_path_modified::String)
        img = load(file)
        img_denoised = denoise_image(img, noise_variance)

        # Maintain the subfolder structure in the new path
        relative_path = relpath(file, base_path)
        save_path = joinpath(base_path_modified, relative_path)
        mkpath(dirname(save_path))

        save(save_path, img_denoised)
    end
end

# Function to perform the main task
function main()
    base_path = "data/CellData/OCT_white_to_black"
    base_path_modified = "data/CellData/OCT_mb3d"
    noise_variance = 0.4
    subfolders = [
        # "DEVELOP"
        "test/NORMAL",
        # "test/DRUSEN",
        # "test/DME",
        # "test/CNV",
    ]

    println("Starting the main task")

    # Flatten the file structure
    all_files = [glob("*.jpeg", joinpath(base_path, subfolder)) for subfolder in subfolders]
    all_files = vcat(all_files...)  # Concatenate all the files into a single array

    # Run the processing in parallel and calculate time for each process
    @sync @distributed for file in all_files
        time_start = time()
        process_file(file, noise_variance, base_path, base_path_modified)
        time_end = time()
        println("Time: $((time_end - time_start)/60)min, File: $file")
    end
end

main()
