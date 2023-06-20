# testing.jl
using Flux
using Flux.Optimise
using Flux: params
using FileIO
using Glob
using CUDA
using JLD2
CUDA.reclaim()
include("VAE_test.jl")
include("data_manipulation/data_loader_MNIST_test.jl")

function train!(model, x, opt, ps)
    batch_loss, back = Flux.pullback(() -> loss(model, x), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end

function main()
    epochs = 1
    load_model_nr = 1
    data_name = "MNIST"
    data_path = "data/MNIST_small"
    save_path = "saved_models/dev.jld2"

    loader = DataLoaderMNIST(data_path, BATCH_SIZE)

    # Initialize CUDA before loading the model
    CUDA.allowscalar(false)
    CUDA.device!(0)  # change 0 to the ID of your GPU, if you have more than one

    if load_model_nr > 0
        # Initialize CUDA before loading the model
        # CUDA.allowscalar(false)
        # CUDA.device!(0)  # change 0 to the ID of your GPU, if you have more than one

        # Create a new model with the same architecture
        # encoder = create_encoder()
        # μ_layer, logvar_layer = create_μ_logvar_layers()
        # decoder = create_decoder()
        # vae = VAE(encoder, μ_layer, logvar_layer, decoder) |> DEVICE

        # # Load the weights into the new model
        # BSON.@load "weights.bson" weights
        # weights = weights |> DEVICE
        # Flux.loadparams!(vae, weights)


        # BSON.@load "model.bson" vae
        vae = load(save_path, "vae")
    else
        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder)
    end

    ps = params(vae)
    opt = ADAM(0.001)

    start_time = time()
    for epoch in 1:epochs
        vae.encoder = vae.encoder |> DEVICE
        vae.μ_layer = vae.μ_layer |> DEVICE
        vae.logvar_layer = vae.logvar_layer |> DEVICE
        vae.decoder = vae.decoder |> DEVICE
        vae = vae |> DEVICE

        println("Epoch: $epoch/$epochs")
        batch_nr = 0
        for (images, labels) in loader
            batch_nr += 1
            if images === nothing
                break
            end
            images = images |> DEVICE
            labels = labels |> DEVICE

            train!(vae, images, opt, ps)
        end

        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

        # Reset the loader for the next epoch
        loader = DataLoaderMNIST(data_path, BATCH_SIZE)

        if load_model_nr > 0
            save_nr = load_model_nr + epoch
        else
            save_nr = epoch
        end
        save_path = "saved_models/$(data_name)_epoch_$(save_nr).jld2"

        # Save the weights of the model
        # weights = Flux.params(vae)
        # weights = cpu(weights)
        # BSON.@save "weights.bson" weights

        # BSON.@save "model.bson" vae
        vae.encoder = vae.encoder |> cpu
        vae.μ_layer = vae.μ_layer |> cpu
        vae.logvar_layer = vae.logvar_layer |> cpu
        vae.decoder = vae.decoder |> cpu
        vae = vae |> cpu
        save(save_path, "vae", vae)
    end
    return nothing
end

main()
