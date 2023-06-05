using Images
using Plots

# Load the data
@load "data/preprocessed_data.jld2" images labels

# Select an image to inspect (assuming data is a matrix of grayscale images)
println(size(images))

image_index = 1
image = images[:, :, image_index]
println(size(image))

# Display the image
heatmap(image, color = :grays, aspect_ratio = :equal, axis = :off)

