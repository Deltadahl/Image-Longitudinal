# data_loader_MNIST.jl
using Images
using FileIO
using Random
using Flux

struct DataLoaderMNIST
    dir::String
    batch_size::Int
    filenames::Vector{String}
end

function DataLoaderMNIST(dir::String, batch_size::Int)
    filenames = readdir(dir)
    Random.shuffle!(filenames)
    return DataLoaderMNIST(dir, batch_size, filenames)
end

function get_label(loader::DataLoaderMNIST, filename::String)
    for i in 0:9
        if startswith(filename, string(i))
            return i
        end
    end
    error("Invalid filename: ", filename)
end

Base.iterate(loader::DataLoaderMNIST, state=1) = state > length(loader.filenames) ? nothing : (next_batch(loader, state), state + loader.batch_size)

function next_batch(loader::DataLoaderMNIST, start_idx::Int)
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
        image = imresize(image, (224, 224))
        # Reshape the image to the format (height, width, channels, batch size)
        image = reshape(image, size(image)..., 1, 1)
        push!(images, image)
        push!(labels, get_label(loader, filename))
    end

    # Concatenate all images along the 4th dimension to form a single batch
    images = cat(images..., dims=4)
    labels = Flux.onehotbatch(labels, 0:9)
    return images, labels
end
