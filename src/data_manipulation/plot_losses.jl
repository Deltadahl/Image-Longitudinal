using Plots

# Load the losses from the files
function load_losses(filename)
    file = open(filename, "r")
    losses = map(line -> parse(Float32, line), readlines(file))
    close(file)
    return losses
end

try_nr = 1
folder_path = "saved_losses/try_$(try_nr)/"

#load the train losses
loss_rec = load_losses(folder_path * "rec.txt")
loss_kl = load_losses(folder_path * "kl.txt")
loss_mse = load_losses(folder_path * "mse.txt")
loss_l2 = load_losses(folder_path * "l2.txt")
loss_l9 = load_losses(folder_path * "l9.txt")

# Load the test losses
loss_rec_test = load_losses(folder_path * "test_rec.txt")
loss_kl_test = load_losses(folder_path * "test_kl.txt")
loss_mse_test = load_losses(folder_path * "test_mse.txt")
loss_l2_test = load_losses(folder_path * "test_l2.txt")
loss_l9_test = load_losses(folder_path * "test_l9.txt")

p1 = plot(loss_rec[1:end], label="Train: Reconstruction loss", lw = 2)
plot!(p1, loss_rec_test, label="Test: Reconstruction loss", lw = 2, linestyle=:dash)

p2 = plot(loss_kl[1:end], label="Train: KL loss", lw = 2)
plot!(p2, loss_kl_test, label="Test: KL loss", lw = 2, linestyle=:dash)

# p3 = plot(loss_mse, label="Train: MSE loss", lw = 2)
# plot!(p3, loss_mse_test, label="Test: MSE loss", lw = 2, linestyle=:dash)

# p4 = plot(loss_l2, label="Train: L2 loss", lw = 2)
# plot!(p4, loss_l2_test, label="Test: L2 loss", lw = 2, linestyle=:dash)

# p5 = plot(loss_l9, label="Train: L9 loss", lw = 2)
# plot!(p5, loss_l9_test, label="Test: L9 loss", lw = 2, linestyle=:dash)

# plot(p1, p2, p3, p4, p5, layout = (5, 1))
plot(p1, p2, layout = (2, 1))


# using Plots

# # Load the losses from the files
# function load_losses(filename)
#     file = open(filename, "r")
#     losses = map(line -> parse(Float32, line), readlines(file))
#     close(file)
#     return losses
# end

# try_nr = 1
# #load the train losses
# loss_rec = load_losses("saved_losses/loss_rec$(try_nr).txt")
# p1 = plot(loss_rec[2:end])
