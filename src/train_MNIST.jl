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

function train!(model, x, opt, ps, y, gl_balance)
    batch_loss, back = Flux.pullback(() -> loss(model, x, y, gl_balance), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end


function main()
    epochs = 100
    load_model = false
    model_name = "MNIST_epoch_30.jld2"
    # data_path = "data/MNIST_small"
    data_path = "data/data_resized/MNIST_small_224"

    loader = DataLoader(data_path, BATCH_SIZE) |> DEVICE

    if load_model
        vae = load("saved_models/" * model_name, "vae") |> DEVICE
    else
        function print_params(model)
            ps = Flux.params(model)
            for (i, p) in enumerate(ps)
                println("Layer $i has $(length(p)) parameters.")
            end
        end

        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder) |> DEVICE

        print_params(vae)
    end

    ps = params(vae)
    opt = ADAM(0.001)

    start_time = time()
    loss_list_rec_saver = []
    loss_list_kl_saver = []
    for epoch in 1:epochs
        gl_balance = GLBalance(0.0f0, 0.0f0)

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

            train!(vae, images, opt, ps, labels, gl_balance)
            println("Batch: $batch_nr")
            println("Loss rec: gl_balance.avg_rec")
            println("Loss kl:  gl_balance.avg_kl")
            println("Loss tot: gl_balance.avg_rec + gl_balance.avg_kl")
        end

        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")


        # rec_loss = sum(loss_list_rec)
        # kl_loss = sum(loss_list_kl)
        rec_loss = gl_balance.avg_rec
        kl_loss = gl_balance.avg_kl
        epoch_loss = rec_loss + kl_loss
        push!(loss_list_rec_saver, rec_loss)
        push!(loss_list_kl_saver, kl_loss)
        println("Loss tot: $(Printf.@sprintf("%.8f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%.8f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.8f", kl_loss))")

        # Reset the loader for the next epoch
        loader.idx = 1
        Random.shuffle!(loader.filenames)

        save_path = "saved_models/MNIST_epoch_$(epoch).jld2"
        save(save_path, "vae", vae)
        println("saved model to $save_path")
    end
    return nothing
end

main()
