# load_and_preprocess.jl
using Images
using ImageMagick
using CSV
using DataFrames
using Flux
using MLDataPattern
using Flux.Data: DataLoader
using Plots
using Shuffle
include("../constants.jl")
# using .Constants

# Define a function to load and preprocess images
function load_and_process_image(file_path)
    img = load(file_path)  # load the image file
    img_gray = Gray.(img)  # convert image to grayscale
    img_gray = Float32.(img_gray)  # convert to Float32
    img = reshape(channelview(img_gray), size(img_gray)..., 1)  # convert to matrix format and add an extra dimension to make it 3D
    return img
end


function get_class_from_filename(file_name)
    parts = split(file_name, "-")
    if length(parts) < 2
        error("File name does not have expected format: $file_name")
    end
    class_str = parts[1]
    if class_str ∉ CLASSES
        println("file_name: $file_name")
        println("class_str: $class_str")
        error("Class from file name is not in CLASSES: $class_str")
    end
    return class_str
end


# Define a function to get one-hot vector from class string
function get_one_hot_from_class(class_str)
    class_index = findfirst(x -> x == class_str, CLASSES)  # get index of the class
    return Flux.onehot(class_index, 1:4)  # convert to one-hot vector
end

# Define a BatchLoader struct
struct BatchLoader
    directory_path::String
    file_paths::Vector{String}
    batch_size::Int64
    current_index::Int64
    num_files::Int64
end

BatchLoader(directory_path::String, batch_size::Int64) = BatchLoader(
    directory_path,
    shuffle(readdir(directory_path)),
    batch_size,
    1,
    length(readdir(directory_path)),
)


Base.length(loader::BatchLoader) = ceil(Int, loader.num_files / loader.batch_size)
Base.getindex(loader::BatchLoader, i::Int) =
    iterate(loader, (i - 1) * loader.batch_size + 1)
Base.getindex(loader::BatchLoader, i::UnitRange) = [getindex(loader, j) for j in i]

function iterate(loader::BatchLoader, state = loader.current_index)
    if state > loader.num_files
        return nothing
    end

    i = state
    batch_file_paths = loader.file_paths[i:min(i + loader.batch_size - 1, end)]
    images = [
        load_and_process_image(joinpath(loader.directory_path, file_path)) for
        file_path in batch_file_paths
    ]  # load and process images
    classes =
        [get_class_from_filename(basename(file_path)) for file_path in batch_file_paths]  # get classes from file names
    labels = [get_one_hot_from_class(class_str) for class_str in classes]  # convert classes to one-hot vectors
    images = cat(images..., dims = 4)

    #return ((images, labels), i + loader.batch_size)
    return images, labels
end


# Define a function to create a data loader
function create_dataloader(directory_path, batch_size)
    return DataLoader(BatchLoader(directory_path, batch_size), batchsize = 1)
end

function print_image(dataloader)
    data = first(dataloader)
    images, labels = data[1]
    # println(size(images))
    # println(size(labels))
    # println(labels[1])
    first_image = images[:, :, 1, 1]  # Extract the first image from the batch

    first_image = Gray.(first_image)  # Convert back to Gray scale
    plot(
        first_image,
        seriestype = :heatmap,
        color = :grays,
        title = "$( CLASSES[labels[1]][1])",
    )
end

function main()
    # Create a dataloader
    dataloader = create_dataloader(DIRECTORY_PATH, BATCH_SIZE)
    print_image(dataloader)
end

@time main()
