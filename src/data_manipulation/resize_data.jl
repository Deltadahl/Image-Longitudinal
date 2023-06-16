using Images
using FileIO

# Set the source and destination directories
src_dir = "data/MNIST_small"
dest_dir = "data/data_resized/MNIST_small_224"

# Create the destination directory if it does not exist
!isdir(dest_dir) && mkdir(dest_dir)

# Loop over all .jpeg files in the source directory
for filename in readdir(src_dir)
    # if endswith(filename, ".jpeg")
    if endswith(filename, ".png")
        # Load the image
        img = load(joinpath(src_dir, filename))

        # Resize the image
        resized_img = imresize(img, (224, 224))

        # Save the resized image to the destination directory
        save(joinpath(dest_dir, filename), resized_img)
    end
end
println("Done!")
