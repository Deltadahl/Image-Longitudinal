# data_loader_OCT.jl
using Images
using FileIO
using Random
using Flux

struct DataLoaderOCT
    dir::String
    batch_size::Int
    filenames::Vector{String}
    data_augmentation::Bool
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

function DataLoaderOCT(dir::String, batch_size::Int, data_augmentation::Bool)
    filenames = readdir(dir)
    # filenames = undersample_dataset(filenames)
    Random.shuffle!(filenames)
    return DataLoaderOCT(dir, batch_size, filenames, data_augmentation)
end

function get_label(loader::DataLoaderOCT, filename::String)
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

# Function to adjust brightness
function adjust_brightness(image, factor)
    return clamp.(image .+ factor, 0, 1)
end

# Function to adjust contrast
function adjust_contrast(image, factor)
    mean_intensity = mean(image)
    return clamp.((image .- mean_intensity) .* factor .+ mean_intensity, 0, 1)
end

Base.iterate(loader::DataLoaderOCT, state=1) = state > length(loader.filenames) ? nothing : (next_batch(loader, state), state + loader.batch_size)

function next_batch(loader::DataLoaderOCT, start_idx::Int)
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
        # Convert the image to grayscale and then to Float32
        image = Float32.(Gray.(image))

        height = 224
        new_height = 190
        if loader.data_augmentation
            # With a 50% chance, mirror the image around the vertical axis
            if rand() < 0.5
                reverse!(image, dims=2)
            end

            # TODO use this when OCT images gives better results
            # r = rand()
            # if r < 0.3
            #     # Generate a random brightness change factor between -0.2 and 0.2
            #     brightness_factor = 0.2 * (rand() - 0.5)
            #     image = adjust_brightness(image, brightness_factor)
            # elseif r < 0.6
            #     # Generate a random contrast change factor between -0.2 and 0.2
            #     contrast_factor = 0.8 + 0.4 * rand() # random contrast factor between 0.8 (decrease) and 1.2 (increase)
            #     image = adjust_contrast(image, contrast_factor)
            # end

            # Generate a random starting height for cropping (the upper bound ensures the cropped portion will fit within the image)
            start_h = rand(1:(height-new_height+1))
        else
            # Without data augmentation, just center crop the image
            start_h = Int(floor((height - new_height) / 2)) + 1
        end
        image = image[start_h:(start_h+new_height-1), :]

        image = imresize(image, (224, 224))
        # Reshape the image to the format (height, width, channels, batch size)
        image = reshape(image, size(image)..., 1, 1)
        push!(images, image)
        push!(labels, get_label(loader, filename))

        # Explicitly delete the image variable to free up memory
        finalize(image) # TODO see if this changes anything
    end

    # Concatenate all images along the 4th dimension to form a single batch
    images = cat(images..., dims=4)
    labels = Flux.onehotbatch(labels, 1:4)
    return images, labels
end
