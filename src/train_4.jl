# train_4.jl
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
    Optimise.update!(opt, params(model), grads)
end

function main()
    # Initialize the data loader
    loader = DataLoader("data/data_resized/all_develop", 64)

    # Create the encoder, mu/logvar layers and decoder
    encoder = create_encoder()
    mu_layer, logvar_layer = create_mu_logvar_layers()
    decoder = create_decoder()

    # Create the VAE
    vae = VAE(encoder, mu_layer, logvar_layer, decoder)

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

            loss_val = loss(images, vae)
            println("Loss for batch $batch_nr: $loss_val")

            train!(vae, images, opt)
        end

        # Reset the loader for the next epoch
        loader.idx = 1
        Random.shuffle!(loader.filenames)
    end

    # Save the model
    @save "vae.bson" vae
end

main()





# # train_4.jl
# using Flux
# using Flux.Optimise
# using Flux: params

# # Include necessary files
# include("data_manipulation/data_loader.jl")
# include("VAE.jl")

# function train!(model, x, opt)
#     grads = Flux.gradient(params(model)) do
#         l = loss(x, model)
#         return l
#     end
#     Optimise.update!(opt, params(model), grads)
# end

# function main()
#     # Initialize the data loader
#     loader = DataLoader("data/data_resized/all_develop", 64)

#     # Create the encoder, mu/logvar layers and decoder
#     encoder = create_encoder()
#     mu_layer, logvar_layer = create_mu_logvar_layers()
#     decoder = create_decoder()

#     # Create the VAE
#     vae = VAE(encoder, mu_layer, logvar_layer, decoder)

#     # Define an optimizer
#     opt = ADAM(0.001)

#     # Number of epochs
#     epochs = 2

#     # Train the model
#     for epoch in 1:epochs
#         println("Epoch: $epoch")
#         batch_nr = 0
#         while true
#             batch_nr += 1
#             @info "Batch $batch_nr"

#             images, _ = next_batch(loader)
#             images = gpu(images)
#             if images === nothing
#                 break
#             end

#             train!(vae, images, opt)
#         end

#         # Reset the loader for the next epoch
#         loader.idx = 1
#         Random.shuffle!(loader.filenames)
#     end
# end

# main()
