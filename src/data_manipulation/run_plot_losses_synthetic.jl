include("plot_losses_synthetic.jl")
try_nr = 13
x_scale = 100_000
images_train = 950_000
plot_losses(try_nr, x_scale, images_train)
println("Done")
