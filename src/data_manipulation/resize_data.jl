using Images
using FileIO

# Set the source and destination directories
src_dir = "data/data_resized/all_train"
dest_dir = "data/data_resized/all_train_256"

# Create the destination directory if it does not exist
!isdir(dest_dir) && mkdir(dest_dir)

# Loop over all .jpeg files in the source directory
for filename in readdir(src_dir)
    if endswith(filename, ".jpeg")
        # Load the image
        img = load(joinpath(src_dir, filename))

        # Resize the image
        resized_img = imresize(img, (256, 256))

        # Save the resized image to the destination directory
        save(joinpath(dest_dir, filename), resized_img)
    end
end
