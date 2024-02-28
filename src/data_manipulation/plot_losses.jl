using Plots

Plots.gr()
# plotlyjs()
function plot_losses(try_nr, evaluate_interval = 5000)
    x_scale = evaluate_interval / IMAGES_TRAIN
    x_scale = 1/floor(Int, 1/x_scale)
    function load_losses(filename)
        file = open(filename, "r")
        losses = map(line -> parse(Float32, line), readlines(file))
        close(file)
        return losses
    end

    folder_path = "saved_losses/try_$(try_nr)/"

    loss_rec = load_losses(folder_path * "rec.txt")
    loss_kl = load_losses(folder_path * "kl.txt")
    loss_mse = load_losses(folder_path * "mse.txt")
    loss_l2 = load_losses(folder_path * "l2.txt")
    loss_l9 = load_losses(folder_path * "l9.txt")

    loss_rec_test = load_losses(folder_path * "test_rec.txt")
    loss_kl_test = load_losses(folder_path * "test_kl.txt")
    loss_mse_test = load_losses(folder_path * "test_mse.txt")
    loss_l2_test = load_losses(folder_path * "test_l2.txt")
    loss_l9_test = load_losses(folder_path * "test_l9.txt")

    # Create total losses
    total_loss = loss_rec .+ loss_kl
    total_loss_test = loss_rec_test .+ loss_kl_test


    # The plot for MSE, L2, and L9 loss
    p3 = plot((1:length(loss_mse))*x_scale, loss_mse, label="Train: MSE loss", lw = 2, linestyle=:dash, color=:blue, leg=:top)
    plot!(p3, (1:length(loss_mse_test))*x_scale, loss_mse_test, label="Test: MSE loss", lw = 2, color=:cyan)
    plot!(p3, (1:length(loss_l2))*x_scale, loss_l2, label="Train: Layer 2 loss", lw = 2, linestyle=:dash, color=:red)
    plot!(p3, (1:length(loss_l2_test))*x_scale, loss_l2_test, label="Test: Layer 2 loss", lw = 2, color=:orange)
    plot!(p3, (1:length(loss_l9))*x_scale, loss_l9, label="Train: Layer 7 loss", lw = 2, linestyle=:dash, color=:green)
    plot!(p3, (1:length(loss_l9_test))*x_scale, loss_l9_test, label="Test: Layer 7 loss", lw = 2, color=:magenta)
    vline!(p3, [5], linestyle=:dot, color=:gray, label="Stopped increasing β", lw=2)
    title!(p3, "MSE, Layer 2, and Layer 7 Loss")
    xlabel!(p3, "Number of Epochs")
    ylabel!(p3, "Loss")
    display(p3)
    save_path = "saved_losses/try_$(try_nr)/losses_L2L7mse.png"
    savefig(p3, save_path)

    # The plot for total loss
    p2 = plot((1:length(total_loss))*x_scale, total_loss[1:end], label="Train: Total loss", lw = 2, linestyle=:dash, color=:blue)
    plot!(p2, (1:length(total_loss_test))*x_scale, total_loss_test, label="Test: Total loss", lw = 2, color=:cyan)
    title!(p2, "Total Loss")
    xlabel!(p2, "Number of Epochs")
    ylabel!(p2, "Loss")
    display(p2)
    # save image
    save_path = "saved_losses/try_$(try_nr)/losses_tot.png"
    savefig(p2, save_path)

    function read_statistics_from_file(filename)
        file = open(filename, "r")
        lines = readlines(file)
        close(file)
        return map(line -> parse(Float32, line), lines)
    end

    # Assuming you've read your data into variables as follows:
    mean_μ = read_statistics_from_file(folder_path * "mean_mu.txt")
    var_μ = read_statistics_from_file(folder_path * "var_mu.txt")
    mean_logvar = read_statistics_from_file(folder_path * "mean_logvar.txt")
    var_logvar = read_statistics_from_file(folder_path * "var_logvar.txt")

    # mean_logvar = exp.(mean_logvar ./ 2)
    # var_σ = exp.(var_logvar ./ 2)


    # Create your plots:
    p5 = plot((1:length(mean_μ))*x_scale, mean_μ, label="Mean of μ", lw = 2, linestyle=:dash, color=:blue, dpi=300, leg=:bottomright)
    # plot!(p5, (1:length(var_μ))*x_scale, var_μ, label="Variance of μ", lw = 2, linestyle=:dash, color=:cyan)
    plot!(p5, (1:length(mean_logvar))*x_scale, mean_logvar, label="Mean of log(σ²)", lw = 2, color=:red, dpi=300)
    # plot!(p5, (1:length(var_logvar))*x_scale, var_logvar, label="Variance of log(σ²)", lw = 2, color=:orange, legs=:top)
    vline!(p5, [0.75], linestyle=:dot, color=:gray, label="Selected Value", lw=2)

    title!(p5, "μ and σ Statistics")
    xlabel!(p5, "Number of Epochs")
    ylabel!(p5, "Value")
    display(p5)
    save_path = "saved_losses/try_$(try_nr)/statistics.png"
    savefig(p5, save_path)

    # The plot for Reconstruction and KL loss
    p1 = plot((1:length(loss_rec))*x_scale, loss_rec[1:end], label="Train: Reconstruction loss", lw = 2, linestyle=:dash, color=:blue)
    plot!(p1, (1:length(loss_rec_test))*x_scale, loss_rec_test, label="Test: Reconstruction loss", lw = 2, color=:cyan)
    plot!(p1, (1:length(loss_kl))*x_scale, loss_kl[1:end], label="Train: KL loss", lw = 2, linestyle=:dash, color=:red)
    plot!(p1, (1:length(loss_kl_test))*x_scale, loss_kl_test, label="Test: KL loss", lw = 2, color=:orange)
    title!(p1, "Reconstruction and KL Loss")
    xlabel!(p1, "Number of Epochs")
    ylabel!(p1, "Loss")
    display(p1)
    save_path = "saved_losses/try_$(try_nr)/losses_kl_rec.png"
    savefig(p1, save_path)

    # The plot for training reconstruction loss starting from x_scale * 2
    # p4 = plot((21:length(loss_rec[21:end]))*x_scale, loss_rec[21:end], label="Train: Reconstruction loss", lw = 2, linestyle=:dash, color=:blue)
    recordings_epoch = floor(Int, IMAGES_TRAIN / evaluate_interval)
    if length(loss_rec) > 2 * recordings_epoch
        p4 = plot((1:length(loss_rec[recordings_epoch:end]))*x_scale .+ 2.0 , loss_rec[recordings_epoch:end], label="Train: Reconstruction loss", lw = 2, color=:blue)
        title!(p4, "Training Reconstruction Loss (Starting from Epoch 2)")
        xlabel!(p4, "Number of Epochs")
        ylabel!(p4, "Loss")
        display(p4)
        # Save image
        save_path = "saved_losses/try_$(try_nr)/losses_rec_from_2.png"
        savefig(p4, save_path)
    end

    p6 = plot((1:length(loss_rec_test))*x_scale, loss_rec_test, label="Reconstruction loss", lw = 2, color=:blue)
    vline!(p6, [5], linestyle=:dot, color=:gray, label="Stopped increasing β", lw=2)
    title!(p6, "Reconstruction Loss Validation Set")
    xlabel!(p6, "Number of Epochs")
    ylabel!(p6, "Loss")
    display(p6)
    save_path = "saved_losses/try_$(try_nr)/losses_rec_test.png"
    savefig(p6, save_path)

    p7 = plot((1:length(loss_kl_test))*x_scale, loss_kl_test, label="KL divergence loss", lw = 2, color=:blue, leg=:bottomright)
    vline!(p7, [5], linestyle=:dot, color=:gray, label="Stopped increasing β", lw=2)
    title!(p7, "KL Divergence Loss Validation Set")
    xlabel!(p7, "Number of Epochs")
    ylabel!(p7, "Loss")
    display(p7)
    save_path = "saved_losses/try_$(try_nr)/losses_kl_test.png"
    savefig(p7, save_path)

    p8 = plot((1:length(total_loss_test))*x_scale, total_loss_test, label="Total loss", lw = 2, color=:blue)
    vline!(p8, [5], linestyle=:dot, color=:gray, label="Stopped increasing β", lw=2)
    title!(p8, "Total Loss Validation Set")
    xlabel!(p8, "Number of Epochs")
    ylabel!(p8, "Loss")
    display(p8)
    save_path = "saved_losses/try_$(try_nr)/losses_total_test.png"
    savefig(p8, save_path)

end
