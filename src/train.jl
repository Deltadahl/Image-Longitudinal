# train.jl
using Flux
using Flux.Optimise
using Flux: params
using BSON: @save

# Include necessary files
include("data_manipulation/data_loader.jl")
include("VAE.jl")

function train!(model, x, opt)
    grads = Flux.gradient(params(model)) do
        l = loss(x, model)
        return l
    end
    # loss_val = loss(x, model)
    # println("Loss: $loss_val")
    Optimise.update!(opt, params(model), grads)
end

function show_image(vae)
    base_path = "data/data_resized/all_develop"
    image_name = "CNV-13823-1.jpeg"
    x = load(joinpath(base_path, image_name))

    # Transform the image to a 4D tensor
    println("1")
    x = reshape(x, (size(x)..., 1, 1)) |> DEVICE
    # image = Float32.(Gray.(image))
    #     # Reshape the image to the format (height, width, channels, batch size)
    #     image = reshape(image, size(image)..., 1, 1)
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

function main()
    # Initialize the data loader
    loader = DataLoader("data/data_resized/all_develop", BATCH_SIZE)

    # Create the encoder, mu/logvar layers and decoder
    encoder = create_encoder()
    mu_layer, logvar_layer = create_mu_logvar_layers()
    decoder = create_decoder()

    # Create the VAE
    vae = VAE(encoder, mu_layer, logvar_layer, decoder) |> DEVICE

    # Define an optimizer
    opt = ADAM(0.001)

    # Number of epochs
    epochs = 2

    # Train the model
    for epoch in 1:epochs
        println("Epoch: $epoch")
        batch_nr = 0
        while true
            batch_nr += 1
            @info "Batch $batch_nr"

            images, _ = next_batch(loader)
            images = gpu(images)
            if images === nothing
                break
            end

            train!(vae, images, opt)
        end

        # Reset the loader for the next epoch
        loader.idx = 1
        Random.shuffle!(loader.filenames)
    end

    # Save the model
    @save "saved_models/vae.bson" vae



end

main()
