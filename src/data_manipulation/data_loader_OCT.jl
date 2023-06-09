# data_loader_OCT.jl
using Images
using FileIO
using Random
using Flux

struct DataLoader
    dir::String
    batch_size::Int
    filenames::Vector{String}
end

function undersample_dataset(filenames)
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
        else
            error("Invalid filename: ", filename)
        end
    end

    # Find the size of the smallest class
    min_size = minimum(length(v) for v in values(classes))

    # Create an array to hold the undersampled filenames
    undersampled_filenames = []

    # Undersample the majority classes
    for (class, class_filenames) in classes
        # Randomly select min_size elements from class_filenames
        random_indices = randperm(length(class_filenames))[1:min_size]
        selected_filenames = class_filenames[random_indices]

        # Add the selected filenames for this class to the undersampled_filenames array
        append!(undersampled_filenames, selected_filenames)
    end

    return undersampled_filenames
end



function DataLoader(dir::String, batch_size::Int)
    filenames = readdir(dir)
    filenames = undersample_dataset(filenames) # TODO add this, and change so that the dataloder is reinitialized after each epoch
    Random.shuffle!(filenames)
    return DataLoader(dir, batch_size, filenames)
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
Base.iterate(loader::DataLoader, state=1) = state > length(loader.filenames) ? nothing : (next_batch(loader, state), state + loader.batch_size)

function next_batch(loader::DataLoader, start_idx)
    end_idx = min(start_idx + loader.batch_size - 1, length(loader.filenames))

    if start_idx > end_idx
        return nothing, nothing
    end

    batch_size = end_idx - start_idx + 1
    images = []
    labels = []

    for i in start_idx:end_idx
        filename = loader.filenames[i]
        image = load(joinpath(loader.dir, filename))

        # With a 50% chance, mirror the image around the vertical axis
        if rand() < 0.5
            reverse!(image, dims=2)
        end

        # Convert the image to grayscale and then to Float32
        image = Float32.(Gray.(image))
        # Reshape the image to the format (height, width, channels, batch size)
        image = reshape(image, size(image)..., 1, 1)
        push!(images, image)
        push!(labels, get_label(filename))

        # Explicitly delete the image variable to free up memory
        finalize(image)
    end

    # loader.idx += batch_size
    # Concatenate all images along the 4th dimension to form a single batch
    images = cat(images..., dims=4)
    labels = Flux.onehotbatch(labels, 1:4)
    return images, labels
end
