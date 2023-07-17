using Plots

# Plots.plotlyjs()
# Plots.gr()
plotlyjs()
function plot_losses(try_nr, evaluate_interval = 5000)
    x_scale = evaluate_interval / IMAGES_TRAIN
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
    p3 = plot((1:length(loss_mse))*x_scale, loss_mse, label="Train: MSE loss", lw = 2, linestyle=:dash, color=:blue)
    plot!(p3, (1:length(loss_mse_test))*x_scale, loss_mse_test, label="Test: MSE loss", lw = 2, color=:cyan)
    plot!(p3, (1:length(loss_l2))*x_scale, loss_l2, label="Train: L2 loss", lw = 2, linestyle=:dash, color=:red)
    plot!(p3, (1:length(loss_l2_test))*x_scale, loss_l2_test, label="Test: L2 loss", lw = 2, color=:orange)
    plot!(p3, (1:length(loss_l9))*x_scale, loss_l9, label="Train: L9 loss", lw = 2, linestyle=:dash, color=:green)
    plot!(p3, (1:length(loss_l9_test))*x_scale, loss_l9_test, label="Test: L9 loss", lw = 2, color=:magenta)
    title!(p3, "MSE, L2, and L9 Loss")
    xlabel!(p3, "Number of Epochs")
    ylabel!(p3, "Loss")
    display(p3)
    save_path = "saved_losses/try_$(try_nr)/losses_L2L9mse.png"
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
    evaluate_interval / IMAGES_TRAIN
    # p4 = plot((21:length(loss_rec[21:end]))*x_scale, loss_rec[21:end], label="Train: Reconstruction loss", lw = 2, linestyle=:dash, color=:blue)
    p4 = plot((1:length(loss_rec[21*2:end]))*x_scale .+ 2.0 , loss_rec[21*2:end], label="Train: Reconstruction loss", lw = 2, linestyle=:dash, color=:blue)
    title!(p4, "Training Reconstruction Loss (Starting from Epoch 2)")
    xlabel!(p4, "Number of Epochs")
    ylabel!(p4, "Loss")
    display(p4)

    # Save image
    save_path = "saved_losses/try_$(try_nr)/losses_rec_from_2.png"
    savefig(p4, save_path)


end
