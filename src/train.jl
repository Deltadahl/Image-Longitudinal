# train.jl
using Flux
using Flux.Optimise
using Flux: params
# using BSON: @save
using JLD2, FileIO
using Glob

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

function main()
    epochs = 3
    load_model = true
    model_name = "epoch_1_batch_END_vae.jld2"

    loader = DataLoader("data/data_resized/all", BATCH_SIZE) |> DEVICE

    if load_model
        vae = load("saved_models/" * model_name, "vae") |> DEVICE
    else
        encoder = create_encoder()
        mu_layer, logvar_layer = create_mu_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, mu_layer, logvar_layer, decoder) |> DEVICE
    end

    opt = ADAM(0.001)

    start_time = time()
    # Train the model
    for epoch in 1:epochs
        println("Epoch: $epoch")
        batch_nr = 0
        while true
            batch_nr += 1

            images, _ = next_batch(loader)
            if images === nothing
                break
            end
            images = images |> DEVICE

            train!(vae, images, opt)
            if batch_nr % 100 == 0
                println("Batch $batch_nr")

                # TODO test to use this instead
                # elapsed_time = time() - start_time  # get elapsed time in seconds
                # hours, rem = divrem(elapsed_time, 3600)  # convert to hours and remainder
                # minutes, seconds = divrem(rem, 60)  # convert remainder to minutes and seconds
                # println("Time elapsed: $(Int(hours))h $(Int(minutes))m $(Int(seconds))s")

                println("Time elapsed: $(time() - start_time)")
                if batch_nr % 1000 == 0
                    save_path = "saved_models/epoch_$(epoch)_batch_$(batch_nr)_vae.jld2"
                    save(save_path, "vae", vae)
                    print("saved model to $save_path")
                end
            end
        end

        # Reset the loader for the next epoch
        loader.idx = 1
        Random.shuffle!(loader.filenames)
        save_path = "saved_models/epoch_$(epochs)_batch_END_vae.jld2"
        save(save_path, "vae", vae)
        print("saved model to $save_path")
    end

    return nothing
end

main()
