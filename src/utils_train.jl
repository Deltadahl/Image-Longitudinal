function train!(model, x, opt, ps, loss_saver, vgg, loss_normalizers, β_nr, statistics_saver, training, epoch)
    batch_loss, back = Flux.pullback(() -> loss(model, x, loss_saver, vgg, loss_normalizers, β_nr, statistics_saver, training, epoch), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end

function get_dataloader(data_name, data_path, batch_size, augment_data=true)
    return DataLoaderOCT(data_path, batch_size, augment_data)
end

function vae_to_device!(vae::VAE, device)
    vae.encoder = vae.encoder |> device
    vae.μ_layer = vae.μ_layer |> device
    vae.logvar_layer = vae.logvar_layer |> device
    vae.decoder = vae.decoder |> device
    vae = vae |> device
end

function save_model(save_nr, vae)
    vae_copy = deepcopy(vae)
    vae_copy = vae_to_device!(vae_copy, cpu)

    save_path = "saved_models/save_nr_$(save_nr).jld2"
    save(save_path, "vae", vae_copy)
    println("saved model to $save_path")
    return nothing
end

function save_losses(loss, filename)
    file = open(filename, "a")
    write(file, "$loss\n")
    close(file)
    nothing
end

function print_and_save(start_time, loss_saver, loss_normalizers, loss_saver_test, loss_normalizers_test, try_nr, save_nr)
    (loss_normalizer_mse, loss_normalizer2, loss_normalizer9) = loss_normalizers
    (loss_normalizer_mse_test, loss_normalizer2_test, loss_normalizer9_test) = loss_normalizers_test

    @show save_nr
    elapsed_time = time() - start_time
    hours, rem = divrem(elapsed_time, 3600)
    minutes, seconds = divrem(rem, 60)
    println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

    rec_loss = loss_saver.avg_rec / loss_saver.counter
    kl_loss = loss_saver.avg_kl / loss_saver.counter
    epoch_loss = rec_loss + kl_loss
    println("Loss tot: $(Printf.@sprintf("%.9f", epoch_loss))\nLoss rec: $(Printf.@sprintf("%.9f", rec_loss))\nLoss kl:  $(Printf.@sprintf("%.9f", kl_loss))")
    mse_loss = loss_normalizer_mse.sum / loss_normalizer_mse.count
    println("Loss MSE: $(Printf.@sprintf("%.9f", mse_loss))")
    l2_loss = loss_normalizer2.sum / loss_normalizer2.count
    println("Loss L2:  $(Printf.@sprintf("%.9f", l2_loss))")
    l9_loss = loss_normalizer9.sum / loss_normalizer9.count
    println("Loss L9:  $(Printf.@sprintf("%.9f", l9_loss))")

    rec_loss_test = loss_saver_test.avg_rec / loss_saver_test.counter
    kl_loss_test = loss_saver_test.avg_kl / loss_saver_test.counter
    epoch_loss_test = rec_loss_test + kl_loss_test
    println("Test losses:")
    println("Loss tot: $(Printf.@sprintf("%.9f", epoch_loss_test))\nLoss rec: $(Printf.@sprintf("%.9f", rec_loss_test))\nLoss kl:  $(Printf.@sprintf("%.9f", kl_loss_test))")
    mse_loss_test = loss_normalizer_mse_test.sum / loss_normalizer_mse_test.count
    println("Loss MSE: $(Printf.@sprintf("%.9f", mse_loss_test))")
    l2_loss_test = loss_normalizer2_test.sum / loss_normalizer2_test.count
    println("Loss L2:  $(Printf.@sprintf("%.9f", l2_loss_test))")
    l9_loss_test = loss_normalizer9_test.sum / loss_normalizer9_test.count
    println("Loss L9:  $(Printf.@sprintf("%.9f", l9_loss_test))")
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
    nothing
end

function reset_normalizers()
    loss_normalizer_mse = LossNormalizer()
    loss_normalizer2 = LossNormalizer()
    loss_normalizer9 = LossNormalizer()
    loss_normalizers = (loss_normalizer_mse, loss_normalizer2, loss_normalizer9)
    loss_saver = LossSaver(0.0f0, 0.0f0, 0.0f0)
    return loss_normalizers, loss_saver
end

function print_layers(model)
    for (i, layer) in enumerate(model)
        println("layer $i: ", repr(layer))
    end
    nothing
end

function print_vae(vae::VAE)
    println("Encoder Layers:")
    print_layers(vae.encoder.layers)
    println("\nμ layer: ", repr(vae.μ_layer))
    println("logvar layer: ", repr(vae.logvar_layer))
    println("\nDecoder Layers:")
    print_layers(vae.decoder.layers)
    nothing
end

mutable struct StepDecay
    initial_lr::Float64
    drop::Float64
    epochs_drop::Int64
end

function (sd::StepDecay)(epoch::Int64)
    lr = sd.initial_lr * sd.drop^((epoch-1)/sd.epochs_drop)
    lr = min(lr, sd.initial_lr * sd.drop^23)
    return lr
end

function save_statistics(statistic, filename)
    file = open(filename, "a")
    write(file, "$statistic\n")
    close(file)
    nothing
end

function print_statistics(statistics_saver, try_nr)
    # mean_μ = statistics_saver.sum_mu / statistics_saver.counter
    # mean_logvar = statistics_saver.sum_logvar / statistics_saver.counter

    # var_μ = statistics_saver.sum_mu2 / statistics_saver.counter - mean_μ^2
    # var_logvar = statistics_saver.sum_logvar2 / statistics_saver.counter - mean_logvar^2

    # println("Mean of μ: $mean_μ, Variance of μ: $var_μ")
    # println("Mean of logvar: $mean_logvar, Variance of logvar: $var_logvar")

    mu_mean_values = mean(statistics_saver.mu_mean_values)
    logvar_mean_values = mean(statistics_saver.logvar_mean_values)
    mu_variance_values = mean(statistics_saver.mu_variance_values)
    logvar_variance_values = mean(statistics_saver.logvar_variance_values)


    folder_name = "saved_losses/try_$(try_nr)/"
    if !isdir(folder_name)
        mkdir(folder_name)
    end

    # Usage
    # save_statistics(mean_μ, folder_name * "mean_mu.txt")
    # save_statistics(var_μ, folder_name * "var_mu.txt")
    # save_statistics(mean_logvar, folder_name * "mean_logvar.txt")
    # save_statistics(var_logvar, folder_name * "var_logvar.txt")
    save_statistics(mu_mean_values, folder_name * "mean_mu.txt")
    save_statistics(logvar_mean_values, folder_name * "mean_logvar.txt")
    save_statistics(mu_variance_values, folder_name * "var_mu.txt")
    save_statistics(logvar_variance_values, folder_name * "var_logvar.txt")

end
