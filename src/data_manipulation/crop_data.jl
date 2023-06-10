using Images
using FileIO
using Glob
using Printf
using PaddedViews

function resize_and_save_image(file::String, new_path::String, new_dims::Tuple{Int,Int})
    # Load the image
    image = load(file)

    # Get the size of the image
    dims = size(image)

    # If the image is larger than the target size, center crop it
    if all(dims .>= new_dims)
        offset = div.((dims .- new_dims), 2)
        image = image[offset[1]+1:offset[1]+new_dims[1], offset[2]+1:offset[2]+new_dims[2]]
        # If the image is smaller than the target size, pad it with white pixels
    elseif any(dims .< new_dims)
        pad_before = div.((new_dims .- dims), 2)
        pad_after = new_dims .- dims .- pad_before
        image = PaddedView(
            Gray(1), # Gray(0) is black
            image,
            (
                pad_before[1] + dims[1] + pad_after[1],
                pad_before[2] + dims[2] + pad_after[2],
            ),
            (pad_before[1] + 1, pad_before[2] + 1),
        )
    end

    # Save the resized/cropped image
    save(new_path, image)
end

# Function to process all images in a directory
function process_images(old_path::String, new_path::String, new_dims::Tuple{Int,Int})
    # Get a list of all .jpeg files in the directory
    files = glob("*.jpeg", old_path)

    # Process all images
    for file in files
        # Construct the new file path
        new_file_path = joinpath(new_path, basename(file))

        # Resize and save the image
        resize_and_save_image(file, new_file_path, new_dims)
    end
end

function main()
    # Call the function for each subdirectory
    base_path = "data/CellData/OCT"
    subfolders = [
        "train/NORMAL",
        "train/DRUSEN",
        "train/DME",
        "train/CNV",
        # "test/NORMAL",
        # "test/DRUSEN",
        # "test/DME",
        # "test/CNV",
    ]
    new_dims = (512, 512) # TODO can change this later

    for subfolder in subfolders
        old_path = joinpath(base_path, subfolder)
        new_path = joinpath("data/data_resized", "all_train_512")

        # Create the new directory if it doesn't exist
        mkpath(new_path)

        @time process_images(old_path, new_path, new_dims)
    end
end
@time main()
