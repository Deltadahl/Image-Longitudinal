using BM3DDenoise
using ImageIO
using FileIO
using LinearAlgebra
using Wavelets
using Images

function denoise_image(img, noise_variance)
    println("1")
    img_array = float(channelview(img))  # Convert image to 2D array
    # img_array = img_array .* 255
    img_denoised = bm3d(img_array, noise_variance)  # Apply BM3D algorithm
    println("4")
    # Normalize the image to be in the range [0, 1]
    # img_denoised = (img_denoised .- minimum(img_denoised)) ./ (maximum(img_denoised) - minimum(img_denoised))
    img_denoised = colorview(Gray, img_denoised)  # Convert back to grayscale image
    return img_denoised
end

function main()
    base_path = "data/CellData/OCT_modified"
    base_path_modified = "data/CellData/OCT_modified_mb3d"
    subfolders = [
        # "test/NORMAL",
        # "test/DRUSEN",
        # "test/DME",
        "test/CNV",
    ]
    noise_variance = 200 / 255  # Use appropriate noise variance for your images
    for subfolder in subfolders
        println("Processing $subfolder")
        directory = joinpath(base_path, subfolder)
        directory_modified = joinpath(base_path_modified, subfolder)
        files = glob("*.jpeg", directory)
        for file in files
            img_path = file  # The 'file' variable already contains the full path
            img = load(img_path)
            img_denoised = denoise_image(img, noise_variance)
            save_path = joinpath(directory_modified, basename(file))
            mkpath(dirname(save_path))
            save(save_path, img_denoised)
        end
    end
end

main()
