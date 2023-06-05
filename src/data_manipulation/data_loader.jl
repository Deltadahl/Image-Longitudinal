using Images
using FileIO
using Random
using Flux
using Plots

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


# function main()
#     CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
#     loader = DataLoader("data/data_resized/all_develop", 64)

#     while true
#         images, labels = next_batch(loader)
#         if images === nothing
#             break
#         end

#         println("image size $(size(images))")
#         println("label size $(size(labels))")
#         first_image = images[:, :, 1, 1]  # Extract the first image from the batch
#         first_image = Gray.(first_image)
#         image_plot = plot(
#             first_image,
#             seriestype = :heatmap,
#             color = :grays,
#             title = "$(CLASSES[labels[:,1]][1])",
#         )
#         # display(image_plot)
#     end
# end

# @time main()
