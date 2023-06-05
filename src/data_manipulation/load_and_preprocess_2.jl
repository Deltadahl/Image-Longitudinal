using Flux: DataLoader
using FileIO
using ImageMagick
using Plots

function load_image(filename)
    return load(filename)
    img_gray = Gray.(img)  # convert image to grayscale
    img_gray = Float32.(img_gray)  # convert to Float32
    img = reshape(channelview(img_gray), size(img_gray)..., 1)  # convert to matrix format and add an extra dimension to make it 3D
    return img
end

function get_label(filename)
    if occursin("CNV", filename)
        return 1
    elseif occursin("DME", filename)
        return 2
    elseif occursin("DRUSEN", filename)
        return 3
    elseif occursin("NORMAL", filename)
        return 4
    else
        error("Invalid filename: $filename")
    end
end

function get_dataset(path)
    filenames = readdir(path)
    images = []
    labels = []
    for (i, filename) in enumerate(filenames)
        push!(images, load_image(joinpath(path, filename)))
        push!(labels, get_label(filename))
        if i % 500 == 0
            @info "Loaded $i images"
        end
    end
    return images, labels
end
function get_data_loader()
    path = "data/data_resized/all"
    # path = "data/data_resized/test/NORMAL"
    images, labels = get_dataset(path)
    data = [(img, lbl) for (img, lbl) in zip(images, labels)]
    dataloader = DataLoader(data, batchsize=32, shuffle=true)
    @info "Data loaded"

    data = first(dataloader)
    images, labels = data[1]
    # println(size(images))
    # println(size(labels))
    # println(labels[1])
    first_image = images[:, :, 1, 1]  # Extract the first image from the batch

    first_image = Gray.(first_image)
    plot(
        first_image,
        seriestype = :heatmap,
        color = :grays,
        title = "$(labels[1])",
    )
    return dataloader
end

function main()
    dataloader = get_data_loader()
    for (i, data) in enumerate(dataloder)
        images, labels = data
        println(size(images))
        println(size(labels))
        println(labels[1])
        first_image = images[:, :, 1, 1]  # Extract the first image from the batch

        first_image = Gray.(first_image)
        plot(
            first_image,
            seriestype = :heatmap,
            color = :grays,
            title = "$(labels[1])",
        )
        break
    end
    return
end
main()
