# train.jl
using Flux
using Flux.Optimise
using Flux: params
# using BSON: @save
using JLD2, FileIO
using Glob

# Include necessary files
# include("data_manipulation/data_loader.jl")
# include("VAE.jl")
include("data_manipulation/data_loader_MNIST.jl")
include("VAE_MNIST.jl")

function train!(model, x, opt)
    grads = Flux.gradient(params(model)) do
        l = loss(x, model)
        return l
    end
    Optimise.update!(opt, params(model), grads)
end

function main()
    epochs = 2
    load_model = false
    vae_mnist = "MNIST"
    model_name = "$(vae_mnist)_epoch_3_batch_END.jld2"
    # data_path = "data/data_resized/all_develop"
    data_path = "data/MNIST"

    loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE

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
        loss_tot = 0.0
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

                elapsed_time = time() - start_time  # get elapsed time in seconds
                hours, rem = divrem(elapsed_time, 3600)  # convert to hours and remainder
                minutes, seconds = divrem(rem, 60)  # convert remainder to minutes and seconds
                println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

                if batch_nr % 1000 == 0
                    save_path = "saved_models/$(vae_mnist)_epoch_$(epoch)_batch_$(batch_nr).jld2"
                    save(save_path, "vae", vae)
                    println("saved model to $save_path")
                end
            end
            # l = loss(images, vae)
        end

        # Reset the loader for the next epoch
        loader.idx = 1
        Random.shuffle!(loader.filenames)
        save_path = "saved_models/$(vae_mnist)_epoch_$(epoch)_batch_END.jld2"
        save(save_path, "vae", vae)
        println("saved model to $save_path")
    end

    return nothing
end

main()
