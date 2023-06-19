# data_loader_MNIST.jl
using Images
using FileIO
using Random
using Flux

mutable struct DataLoader
    dir::String
    batch_size::Int
    filenames::Vector{String}
    idx::Int
end

function DataLoader(dir::String, batch_size::Int)
    filenames = readdir(dir)
    Random.shuffle!(filenames)
    return DataLoader(dir, batch_size, filenames, 1)
end

function get_label(filename::String)
    for i in 0:9
        if startswith(filename, string(i))
            return i
        end
    end
    error("Invalid filename: ", filename)
end

function next_batch(loader::DataLoader)
    start_idx = loader.idx
    end_idx = min(loader.idx + loader.batch_size - 1, length(loader.filenames))

    if start_idx >= end_idx
        return nothing, nothing
    end

    batch_size = end_idx - start_idx + 1
    images = []
    labels = []

    for i in 1:batch_size
        filename = loader.filenames[start_idx + i - 1]
        image = load(joinpath(loader.dir, filename))
        # Convert the image to grayscale and then to Float32
        image = Float32.(Gray.(image))
        image = imresize(image, (224, 224)) # TODO JUST DEVELOPING....

        # Reshape the image to the format (height, width, channels, batch size)
        image = reshape(image, size(image)..., 1, 1)
        # image = repeat(image, outer=(1,1,3)) # TODO JUST DEVELOPING....
        push!(images, image)
        push!(labels, get_label(filename))
    end

    loader.idx += batch_size
    # Concatenate all images along the 4th dimension to form a single batch
    images = cat(images..., dims=4)
    labels = Flux.onehotbatch(labels, 0:9)
    return images, labels
end
