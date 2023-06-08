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

function train!(model, x, opt, ps, y, loss_list_rec, loss_list_kl)
    batch_loss, back = Flux.pullback(() -> loss(model, x, y, loss_list_rec, loss_list_kl), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end


function main()
    epochs = 20
    load_model = false
    model_name = "MNIST_epoch_3_batch_END.jld2"
    data_path = "data/MNIST"

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
    loss_list_rec_saver = []
    loss_list_kl_saver = []
    for epoch in 1:epochs
        loss_list_rec = []
        loss_list_kl = []
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

            train!(vae, images, opt, ps, labels, loss_list_rec, loss_list_kl)
        end

        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")


        rec_loss = sum(loss_list_rec)
        kl_loss = sum(loss_list_kl)
        epoch_loss = rec_loss + kl_loss
        push!(loss_list_rec_saver, rec_loss)
        push!(loss_list_kl_saver, kl_loss)
        println("Loss tot: $(Printf.@sprintf("%.4f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%.4f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.4f", kl_loss))")

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
