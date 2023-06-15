using Distributed

# Add the number of cores to use
addprocs(10)

@everywhere begin
    using BM3DDenoise
    using ImageIO
    using FileIO
    using Images
    using Glob
    using Printf

    function denoise_image(img::AbstractArray, noise_variance::Float64)::AbstractArray
        img_array = float(channelview(img))
        img_denoised = bm3d(img_array, noise_variance)
        img_denoised = (img_denoised .- minimum(img_denoised)) ./ (maximum(img_denoised) - minimum(img_denoised))
        img_denoised = colorview(Gray, img_denoised)
        return img_denoised
    end

    function process_file(file::String, noise_variance::Float64, base_path::String, base_path_modified::String)
        relative_path = relpath(file, base_path)
        save_path = joinpath(base_path_modified, replace(relative_path, ".jpeg" => "_noise_$noise_variance.jpeg"))

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
    base_path_modified = "data/Noise_testing"
    noise_variances = 0.01:0.01:0.15
    subfolders = [
        "train/DRUSEN",
    ]

    println("Starting the main task")
    all_files = [glob("*.jpeg", joinpath(base_path, subfolder)) for subfolder in subfolders]
    all_files = sort(vcat(all_files...))  # Sort and concatenate all the files into a single array

    file = all_files[20]


    @sync @distributed for noise_variance in noise_variances
        process_file(file, noise_variance, base_path, base_path_modified)
        println("Finished processing $file with noise variance $noise_variance")
    end
end

# Calculate the total time and start the main task
total_time_start = time()
main()
total_time_end = time()
elapsed_total_time = (total_time_end - total_time_start) / 3600  # convert to minutes
println("Total time for all processes: ", @sprintf("%.2f", elapsed_total_time), " hours")
