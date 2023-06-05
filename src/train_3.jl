using Flux
using Flux: @epochs, params
include("constants.jl")
include("data_manipulation/load_and_preprocess.jl")
include("VAE.jl")

function train!(model, dataloader, opt)
    ps = params(model)
    for data in dataloader
        println("Training on batch...")
        x, y = data[1]
        gs = gradient(ps) do
            decoded, mu, logvar = model(x)
            reconstruction_loss = mse(decoded, x)
            kl_divergence = -0.5 .* sum(1 .+ logvar .- mu .^ 2 .- exp.(logvar))
            total_loss = reconstruction_loss + kl_divergence
            return total_loss
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
end

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
        train!(model, dataloader, opt)
        @info "Epoch $epoch completed"
    end
end

function main()
    @info "Starting training..."
    train_vae(100)  # Train for X epochs
    @info "Training complete."
end

main()
