# train.jl
using Flux
using Flux.Optimise
using Flux: params
using JLD2, FileIO
using Glob
using Printf
using ProgressMeter
include("data_manipulation/data_loader_MNIST.jl")
include("VAE_MNIST.jl")

function train!(model, x, opt, ps, y)
    batch_loss, back = Flux.pullback(() -> loss(model, x, y), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    return batch_loss
end


function main()
    epochs = 30
    load_model = false
    model_name = "MNIST_epoch_3_batch_END.jld2"
    data_path = "data/MNIST_small"

    loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE

    if load_model
        vae = load("saved_models/" * model_name, "vae") |> DEVICE
    else
        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder) |> DEVICE
    end

    ps = params(vae)
    opt = ADAM(0.001)

    start_time = time()
    loss_list = []

    for epoch in 1:epochs
        epoch_loss = 0.0
        println("Epoch: $epoch/$epochs")
        batch_nr = 0
        while true
            batch_nr += 1

            images, labels = next_batch(loader)
            if images === nothing
                break
            end
            images = images |> DEVICE
            labels = labels |> DEVICE

            batch_loss = train!(vae, images, opt, ps, labels)
            epoch_loss += batch_loss
        end

        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

        push!(loss_list, epoch_loss)
        println("Loss: $(Printf.@sprintf("%.4f", epoch_loss))")

        # Reset the loader for the next epoch
        loader.idx = 1
        Random.shuffle!(loader.filenames)
        save_path = "saved_models/MNIST_epoch_$(epoch)_batch_END.jld2"
        save(save_path, "vae", vae)
        println("saved model to $save_path")
    end

    return nothing
end

main()
