# train.jl
using Flux
using Flux.Optimise
using Flux: params
using JLD2, FileIO
using Glob
using Printf
using CUDA
include("data_manipulation/data_loader_MNIST.jl")
include("data_manipulation/data_loader_OCT.jl")
include("VAE.jl")
CUDA.math_mode!(CUDA.PEDANTIC_MATH)

function train!(model, x, opt, ps, y, loss_saver, vgg, loss_normalizers)
    batch_loss, back = Flux.pullback(() -> loss(model, x, y, loss_saver, vgg, loss_normalizers), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end

function get_dataloader(data_name, data_path, batch_size)
    if data_name == "OCT"
        return DataLoaderOCT(data_path, batch_size, true)
    else
        return DataLoaderMNIST(data_path, batch_size) #|> DEVICE
    end
end

function main()
    epochs = 1
    load_model_nr = 1
    data_name = "MNIST"
    data_path = "data/MNIST_small"
    # data_name = "OCT"
    # data_path = "data/data_resized/bm3d_496_512_train"

    model_name = "$(data_name)_epoch_$(load_model_nr).jld2"

    loader = get_dataloader(data_name, data_path, BATCH_SIZE)

    if load_model_nr > 0
        vae = load("saved_models/" * model_name, "vae") |> DEVICE
    else
        # function print_params(model)
        #     ps = Flux.params(model)
        #     for (i, p) in enumerate(ps)
        #         println("Layer $i has $(length(p)) parameters.")
        #     end
        # end

        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder) |> DEVICE

        # print_params(vae)
    end

    ps = params(vae)
    opt = ADAM(0.001)

    vgg = vgg_subnetworks()

    start_time = time()
    loss_list_rec_saver = []
    loss_list_kl_saver = []
    for epoch in 1:epochs
        loss_normalizer_mse = LossNormalizer()
        loss_normalizer2 = LossNormalizer()
        loss_normalizer9 = LossNormalizer()
        loss_normalizers = [loss_normalizer_mse, loss_normalizer2, loss_normalizer9]
        loss_saver = LossSaver(0.0f0, 0.0f0, 0.0f0)

        println("Epoch: $epoch/$epochs")
        batch_nr = 0 # TODO enumerate
        for (images, labels) in loader
            batch_nr += 1
            if images === nothing
                break
            end
            images = images |> DEVICE
            labels = labels |> DEVICE

            train!(vae, images, opt, ps, labels, loss_saver, vgg, loss_normalizers)
        end

        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")


        # rec_loss = sum(loss_list_rec)
        # kl_loss = sum(loss_list_kl)
        rec_loss = loss_saver.avg_rec / loss_saver.counter
        kl_loss = loss_saver.avg_kl / loss_saver.counter
        epoch_loss = rec_loss + kl_loss
        push!(loss_list_rec_saver, rec_loss)
        push!(loss_list_kl_saver, kl_loss)
        println("Loss tot: $(Printf.@sprintf("%.8f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%.8f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.8f", kl_loss))")

        # Reset the loader for the next epoch
        loader = get_dataloader(data_name, data_path, BATCH_SIZE)

        if load_model_nr > 0
            save_nr = load_model_nr + epoch
        else
            save_nr = epoch
        end
        save_path = "saved_models/$(data_name)_epoch_$(save_nr).jld2"
        save(save_path, "vae", vae)
        println("saved model to $save_path")
    end
    return nothing
end

main()
