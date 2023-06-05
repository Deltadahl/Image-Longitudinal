# load_model.jl
using BSON: @load
using Images

include("VAE.jl")

function main()
    # Load the model
    @load "saved_models/vae.bson" vae
    vae = vae |> DEVICE
    # Load an image
    base_path = "data/data_resized/all_develop"
    image_name = "CNV-13823-1.jpeg"
    x = load(joinpath(base_path, image_name))

    # Transform the image to a 4D tensor
    println("1")

    x = Float32.(Gray.(x))
        # Reshape the image to the format (height, width, channels, batch size)
    x = reshape(x, size(x)..., 1, 1)
    x = x |> DEVICE

    println("2")
    println("x size $(size(x))")
    # Run the VAE on the image
    reconstructed, _, _ = vae(x)
    println("3")
    # Convert the reconstructed tensor back to an image
    # reconstructed_image = Images.colorview(Gray, reconstructed)
    reconstructed = cpu(reconstructed)
    reconstructed_image = Images.colorview(Gray, dropdims(reconstructed, dims=3))  # remove the singleton dimensions

    # Save the reconstructed image
    save("reconstructed_image.png", reconstructed_image)
end

main()
