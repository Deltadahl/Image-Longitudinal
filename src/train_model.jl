# train.jl
using Flux
using Flux.Optimise
using Flux: params
using JLD2, FileIO
using Glob
using Printf
using CUDA
include("VAE.jl")
include("data_manipulation/data_loader_OCT.jl")
include("generate_image.jl")

function train!(model, x, opt, ps, loss_saver, vgg, loss_normalizers, epoch)
    batch_loss, back = Flux.pullback(() -> loss(model, x, loss_saver, vgg, loss_normalizers, epoch), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end

function get_dataloader(data_name, data_path, batch_size, augment_data=true)
    return DataLoaderOCT(data_path, batch_size, augment_data)
end

function save_model(data_name, save_nr, vae)
    vae.encoder = vae.encoder |> cpu
    vae.μ_layer = vae.μ_layer |> cpu
    vae.logvar_layer = vae.logvar_layer |> cpu
    vae.decoder = vae.decoder |> cpu
    vae = vae |> cpu

    save_path = "saved_models/$(data_name)_epoch_$(save_nr).jld2"
    save(save_path, "vae", vae)
    println("saved model to $save_path")

    vae.encoder = vae.encoder |> DEVICE
    vae.μ_layer = vae.μ_layer |> DEVICE
    vae.logvar_layer = vae.logvar_layer |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE
end

function save_losses(losses, filename)
    # concat filename with folder "saved_losses"
    file = open(filename, "a") # open file in append mode
    for loss in losses
        write(file, "$loss\n") # write each loss on a new line
    end
    close(file) # close the file
end

function main()
    epochs = 100000
    load_model_nr = 0
    try_nr = 4

    data_name = "OCT"
    data_path = "data/data_resized/bm3d_224_train"
    data_path_test = "data/data_resized/bm3d_224_test"
    # data_path = "data/data_resized/bm3d_496_512_test"

    model_name = "$(data_name)_epoch_$(load_model_nr).jld2"

    loader = get_dataloader(data_name, data_path, BATCH_SIZE, true)
    loader_test = get_dataloader(data_name, data_path_test, BATCH_SIZE, false)

    if load_model_nr > 0
        vae = load("saved_models_vgg/" * model_name, "vae") # TODO _vgg change...
    else
        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder)
    end

    function print_layers(model)
        for (i, layer) in enumerate(model)
            println("layer $i: ", repr(layer))
        end
    end

    function print_vae(vae::VAE)
        println("Encoder Layers:")
        print_layers(vae.encoder.layers)
        println("\nμ layer: ", repr(vae.μ_layer))
        println("logvar layer: ", repr(vae.logvar_layer))
        println("\nDecoder Layers:")
        print_layers(vae.decoder.layers)
    end

    # To print the VAE structure:
    print_vae(vae)

    vae.encoder = vae.encoder |> DEVICE
    vae.μ_layer = vae.μ_layer |> DEVICE
    vae.logvar_layer = vae.logvar_layer |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE

    ps = params(vae)
    η₀ = 0.001  # initial learning rate
    decay = 0.95
    opt = ADAM(η₀)

    vgg = vgg_subnetworks()
    # vgg = nothing

    start_time = time()
    for epoch in 1:epochs
        # opt.eta = η₀ * decay^(epoch-1)
        if load_model_nr > 0
            save_nr = load_model_nr + epoch
        else
            save_nr = epoch
        end
        loss_normalizer_mse = LossNormalizer()
        loss_normalizer2 = LossNormalizer()
        loss_normalizer9 = LossNormalizer()
        loss_normalizer_encoded = LossNormalizer()
        loss_normalizers = [loss_normalizer_mse, loss_normalizer2, loss_normalizer9, loss_normalizer_encoded]
        loss_saver = LossSaver(0.0f0, 0.0f0, 0.0f0)

        loss_normalizer_mse_test = LossNormalizer()
        loss_normalizer2_test = LossNormalizer()
        loss_normalizer9_test = LossNormalizer()
        loss_normalizer_encoded_test = LossNormalizer()
        loss_normalizers_test = [loss_normalizer_mse_test, loss_normalizer2_test, loss_normalizer9_test, loss_normalizer_encoded_test]
        loss_saver_test = LossSaver(0.0f0, 0.0f0, 0.0f0)

        println("Epoch: $epoch/$epochs")
        batch_nr = 0 # TODO enumerate
        for (images, _) in loader
            batch_nr += 1
            if images === nothing
                break
            end
            images = images |> DEVICE

            train!(vae, images, opt, ps, loss_saver, vgg, loss_normalizers, save_nr)
        end

        for (images, _) in loader_test
            if images === nothing
                break
            end
            images = images |> DEVICE

            loss(vae, images, loss_saver_test, vgg, loss_normalizers_test, save_nr)
        end

        elapsed_time = time() - start_time
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

        rec_loss = loss_saver.avg_rec / loss_saver.counter
        kl_loss = loss_saver.avg_kl / loss_saver.counter
        epoch_loss = rec_loss + kl_loss
        println("Loss tot: $(Printf.@sprintf("%.5f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%.5f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.5f", kl_loss))")
        mse_loss = loss_normalizer_mse.sum / loss_normalizer_mse.count
        println("Loss MSE: $(Printf.@sprintf("%.5f", mse_loss))")
        l2_loss = loss_normalizer2.sum / loss_normalizer2.count
        println("Loss L2:  $(Printf.@sprintf("%.5f", l2_loss))")
        l9_loss = loss_normalizer9.sum / loss_normalizer9.count
        println("Loss L9:  $(Printf.@sprintf("%.5f", l9_loss))")

        rec_loss_test = loss_saver_test.avg_rec / loss_saver_test.counter
        kl_loss_test = loss_saver_test.avg_kl / loss_saver_test.counter
        epoch_loss_test = rec_loss_test + kl_loss_test
        println("Test losses:")
        println("Loss tot: $(Printf.@sprintf("%.5f", epoch_loss_test))\nLoss rec: $(Printf.@sprintf("%.5f", rec_loss_test))\nLoss kl:  $(Printf.@sprintf("%.5f", kl_loss_test))")
        mse_loss_test = loss_normalizer_mse_test.sum / loss_normalizer_mse_test.count
        println("Loss MSE: $(Printf.@sprintf("%.5f", mse_loss_test))")
        l2_loss_test = loss_normalizer2_test.sum / loss_normalizer2_test.count
        println("Loss L2:  $(Printf.@sprintf("%.5f", l2_loss_test))")
        l9_loss_test = loss_normalizer9_test.sum / loss_normalizer9_test.count
        println("Loss L9:  $(Printf.@sprintf("%.5f", l9_loss_test))")
        println()


        folder_name = "saved_losses/try_$(try_nr)/"
        if !isdir(folder_name)
            mkdir(folder_name)
        end
        # Save losses to file
        save_losses(rec_loss, folder_name * "rec.txt")
        save_losses(kl_loss, folder_name * "kl.txt")
        save_losses(mse_loss, folder_name * "mse.txt")
        save_losses(l2_loss, folder_name * "l2.txt")
        save_losses(l9_loss, folder_name * "l9.txt")

        save_losses(rec_loss_test, folder_name * "test_rec.txt")
        save_losses(kl_loss_test, folder_name * "test_kl.txt")
        save_losses(mse_loss_test, folder_name * "test_mse.txt")
        save_losses(l2_loss_test, folder_name * "test_l2.txt")
        save_losses(l9_loss_test, folder_name * "test_l9.txt")

        # Reset the loader for the next epoch
        loader = get_dataloader(data_name, data_path, BATCH_SIZE, true)
        loader_test = get_dataloader(data_name, data_path_test, BATCH_SIZE, false)

        save_model(data_name, save_nr, vae)
        output_image(vae, loader; epoch=save_nr) # TODO REMOVE OR USE NEW LOADER, THIS uses up the first batch.
    end
    return nothing
end

main()
