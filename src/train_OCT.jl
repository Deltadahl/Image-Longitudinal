# train_OCT.jl
using Flux
using Flux.Optimise
using Flux: params
using JLD2, FileIO
using Glob
using Printf
using ProgressMeter
using CUDA
include("data_manipulation/data_loader_OCT.jl")
include("VAE_OCT.jl")

function train!(model, x, opt, ps, y, loss_list_rec, loss_list_kl)
    batch_loss, back = Flux.pullback(() -> loss(model, x, y, loss_list_rec, loss_list_kl), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end


function main()
    epochs = 3

    load_model = true
    load_model_nr = 1
    model_name = "OCT_epoch_$load_model_nr.jld2"
    data_path = "data/data_resized/all_train_256"

    # loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE
    loader = DataLoader(data_path, BATCH_SIZE)

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
        for (images, labels) in loader
            batch_nr += 1
            if images === nothing
                break
            end
            images = images |> DEVICE
            labels = labels |> DEVICE

            train!(vae, images, opt, ps, labels, loss_list_rec, loss_list_kl)

            Δbatch = 100
            if batch_nr % Δbatch == 0
                println("Batch: $batch_nr")
                list_len = length(loss_list_rec)
                rec_loss = sum(loss_list_rec[list_len-Δbatch+1:end]) / Δbatch
                kl_loss = sum(loss_list_kl[list_len-Δbatch+1:end]) / Δbatch
                epoch_loss = rec_loss + kl_loss
                println("Loss tot: $(Printf.@sprintf("%.6f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%6f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.6f", kl_loss))")
                # CUDA.reclaim() # TODO test if this helps with memory
            end

            # CUDA.reclaim()
        end
        println("--- Epoch $(epoch) finished ---")
        println()
        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")


        rec_loss = sum(loss_list_rec) / length(loss_list_rec)
        kl_loss = sum(loss_list_kl) / length(loss_list_kl)
        epoch_loss = rec_loss + kl_loss
        push!(loss_list_rec_saver, rec_loss)
        push!(loss_list_kl_saver, kl_loss)
        println("Loss tot: $(Printf.@sprintf("%.6f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%.6f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.6f", kl_loss))")

        # Reset the loader for the next epoch
        loader = DataLoader(data_path, BATCH_SIZE)

        if load_model
            save_nr = load_model_nr + epoch
        else
            save_nr = epoch
        end
        save_path = "saved_models/OCT_epoch_$(save_nr).jld2"
        save(save_path, "vae", vae)
        println("saved model to $save_path")
        println()
        CUDA.reclaim()
    end

    return nothing
end

main()
