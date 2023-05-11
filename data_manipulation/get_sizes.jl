using Images
using FileIO
using Glob
using Printf
using DataStructures

function get_image_sizes(path::String, extension::String)
    # Get a list of all .jpeg files in the directory
    files = glob("*.$extension", path)

    # Initialize a dictionary to store the image sizes and their counts
    image_sizes = DefaultDict{Tuple{Int64, Int64}, Int64}(0)

    # Loop over the files
    for file in files
        # Load the image
        image = load(file)

        # Get the size of the image and increment its count in the dictionary
        image_sizes[size(image)] += 1
    end

    return image_sizes
end

function print_image_sizes(label::String, image_sizes::DefaultDict{Tuple{Int64, Int64}, Int64})
    # Convert the dictionary to an array of pairs and sort it by y dimension first, then x dimension
    image_sizes = sort(collect(image_sizes), by = x -> (x[1][2], x[1][1]))

    # Print the sorted image sizes and their counts
    println("Statistics for $label")
    for (size, count) in image_sizes
        @printf("Image size: %d x %d, count: %d\n", size[1], size[2], count)
    end
    println("\n")
end

function main()
    # Call the functions
    base_path = "CellData/OCT"
    subfolders = ["train/NORMAL", "train/DRUSEN", "train/DME", "train/CNV", "test/NORMAL", "test/DRUSEN", "test/DME", "test/CNV"]
    total_image_sizes = DefaultDict{Tuple{Int64, Int64}, Int64}(0)

    for subfolder in subfolders
        path_current = joinpath(base_path, subfolder)
        image_sizes = get_image_sizes(path_current, "jpeg")
        print_image_sizes(subfolder, image_sizes)
        for (size, count) in image_sizes
            total_image_sizes[size] += count
        end
    end

    print_image_sizes("Total", total_image_sizes)
end

@time main()
