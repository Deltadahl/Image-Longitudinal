using Plots

# Plots.gr()
plotlyjs()

function plot_losses(try_nr, evaluate_interval = 5000, images_train=950000)
    x_scale = evaluate_interval / images_train
    x_scale = 1/floor(Int, 1/x_scale)

    function load_losses(filename)
        file = open(filename, "r")
        losses = map(line -> parse(Float32, line), readlines(file))
        close(file)
        return losses
    end

    folder_path = "synthetic_saved_losses/try_$(try_nr)/"

    loss_train = load_losses(folder_path * "loss_train.txt")
    loss_test = load_losses(folder_path * "loss_test.txt")


    # The plot for total loss
    p2 = plot((1:length(loss_train))*x_scale, loss_train[1:end], label="Train loss", lw = 2, linestyle=:dash, color=:blue)
    plot!(p2, (1:length(loss_test))*x_scale, loss_test, label="Test loss", lw = 2, color=:cyan)
    # title!(p2, "Loss")
    xlabel!(p2, "Number of Epochs")
    ylabel!(p2, "Loss")
    display(p2)
    # save image
    save_path = "synthetic_saved_losses/try_$(try_nr)/losses_tot.png"
    savefig(p2, save_path)
end
