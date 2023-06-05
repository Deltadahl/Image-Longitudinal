# train_2.jl
using Flux
using Flux: @epochs, params
include("constants.jl")
include("data_manipulation/load_and_preprocess.jl")
include("VAE.jl")

function train_vae(num_epochs)
    # Load the data
    dataloader = create_dataloader(DIRECTORY_PATH, BATCH_SIZE)

    # Create the model
    encoder = create_encoder()
    mu_layer, logvar_layer = create_mu_logvar_layers()
    decoder = create_decoder()
    model = VAE(encoder, mu_layer, logvar_layer, decoder)

    # Create the optimizer
    opt = ADAM()

    # Training loop
    for epoch in 1:num_epochs
        @info "Epoch $epoch"
        for data in dataloader
            images, labels = data[1][1]
            # TODO ...
        end
    end
end


function main()
    @info "Starting training..."
    train_vae(1)  # Train for X epochs
    @info "Training complete."
end

main()
