using Images
using FileIO
using Random
using Flux
# using Plots

mutable struct DataLoader
    dir::String
    batch_size::Int
    filenames::Vector{String}
    idx::Int
end

function augment_dataset(filenames)
    # Create a dictionary to hold the filenames for each class
    classes = Dict("CNV" => [], "DME" => [], "DRUSEN" => [], "NORMAL" => [])

    # Assign filenames to their respective classes
    for filename in filenames
        if startswith(filename, "CNV")
            push!(classes["CNV"], filename)
        elseif startswith(filename, "DME")
            push!(classes["DME"], filename)
        elseif startswith(filename, "DRUSEN")
            push!(classes["DRUSEN"], filename)
        elseif startswith(filename, "NORMAL")
            push!(classes["NORMAL"], filename)
        end
    end

    # Find the size of the largest class
    max_size = maximum(length(v) for v in values(classes))

    # Create an array to hold the augmented filenames
    augmented_filenames = []

    # Upsample the minority classes
    for (class, filenames) in classes
        while length(filenames) < max_size
            # Duplicate filenames from the start of the array
            push!(filenames, filenames...)
        end

        # If the number of filenames is now greater than max_size, truncate the array
        if length(filenames) > max_size
            filenames = filenames[1:max_size]
        end

        # Add the filenames for this class to the augmented_filenames array
        append!(augmented_filenames, filenames)
    end

    return augmented_filenames
end


function DataLoader(dir::String, batch_size::Int)
    filenames = readdir(dir)
    filenames = augment_dataset(filenames)
    Random.shuffle!(filenames)
    return DataLoader(dir, batch_size, filenames, 1)
end

function get_label(filename::String)
    if startswith(filename, "CNV")
        return 1
    elseif startswith(filename, "DME")
        return 2
    elseif startswith(filename, "DRUSEN")
        return 3
    elseif startswith(filename, "NORMAL")
        return 4
    else
        error("Invalid filename: ", filename)
    end
end

function next_batch(loader::DataLoader)
    start_idx = loader.idx
    end_idx = min(loader.idx + loader.batch_size - 1, length(loader.filenames))

    if start_idx > end_idx
        return nothing, nothing
    end

    batch_size = end_idx - start_idx + 1
    images = []
    labels = []

    for i in 1:batch_size
        filename = loader.filenames[start_idx + i - 1]
        image = load(joinpath(loader.dir, filename))

        # With a 50% chance, mirror the image around the vertical axis
        if rand() < 0.5
            image = flipdim(image, 2)
        end

        # Convert the image to grayscale and then to Float32
        image = Float32.(Gray.(image))
        # Reshape the image to the format (height, width, channels, batch size)
        image = reshape(image, size(image)..., 1, 1)
        push!(images, image)
        push!(labels, get_label(filename))
    end

    loader.idx += batch_size
    # Concatenate all images along the 4th dimension to form a single batch
    images = cat(images..., dims=4)
    labels = Flux.onehotbatch(labels, 1:4)
    return images, labels
end
