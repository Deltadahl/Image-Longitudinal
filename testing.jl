using Images, FileIO

function read_mnist_images(filename)
    f = open(filename)
    magic_number = ntoh(read(f, Int32))
    num_images = ntoh(read(f, Int32))
    num_rows = ntoh(read(f, Int32))
    num_cols = ntoh(read(f, Int32))
    images = Array{UInt8, 2}[]
    for i in 1:num_images
        image = reshape(read(f, UInt8, num_rows*num_cols), num_rows, num_cols)
        push!(images, image)
    end
    close(f)
    return images
end

function save_images(images, folder)
    for (i, image) in enumerate(images)
        save(joinpath(folder, "$i.png"), Gray.(image ./ 255))
    end
end

# Replace 'train-images-idx3-ubyte' with the path to your file
images = read_mnist_images("data/train-images.idx3-ubyte")
save_images(images, "output_folder")
