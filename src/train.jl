using Flux
using Flux.Optimise: update!
include("data_manipulation/load_and_preprocess.jl")
include("VAE.jl")

function train_vae()
    # Load data
    dataloader = create_dataloader(DIRECTORY_PATH, BATCH_SIZE)

    # Instantiate the VAE
    encoder = create_encoder()
    mu_layer, logvar_layer = create_mu_logvar_layers()
    decoder = create_decoder()
    model = VAE(encoder, mu_layer, logvar_layer, decoder)

    # Define the optimizer
    opt = ADAM()

    # Number of training epochs
    num_epochs = 1

    # Training loop
    for epoch in 1:num_epochs
        total_loss = 0

        for data in dataloader
            x, y = data[1][1]
            # println("x: $x, y: $y")
            x, y = x |> DEVICE, y |> DEVICE  # Move data to device

            # Compute gradients
            grads = gradient(Flux.params(model)) do
                l = loss(x, model)
                total_loss += l
                return l
                println("Loss: $l")
            end

            # Update weights
            for p in Flux.params(model)
                update!(opt, p, grads[p])
            end
        end

        avg_loss = total_loss / length(dataloader)
        println("Epoch: $epoch, Loss: $avg_loss")
    end

    # Save the model
    Flux.save("vae.bson", model)
end

train_vae()
