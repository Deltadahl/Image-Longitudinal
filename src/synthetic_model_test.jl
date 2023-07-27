# synthetic_train_model.jl
using JLD2, FileIO
using Flux
using Flux.Optimise
using Flux: params
using Glob
using Printf
using CUDA
using Flux.Data: DataLoader
include("synthetic_model.jl")
include("synthetic_utils_train.jl")
include("data_manipulation/plot_losses_synthetic.jl")
include("VAE.jl")

function main()
    load_model_nr = 45
    load_model_nr_vae = 526
    try_nr = 8

    batch_size = 24

    filepath = "data/synthetic/eta_and_lv_data_1_new.jld2"
    η_matrix = load(filepath, "eta_matrix")
    η_matrix = Float32.(η_matrix)
    lvs_matrix = load(filepath, "lvs_matrix")

    # mse lvs_matrix 92, 111 and 50 vs η_matrix
    @show mean((lvs_matrix[92, :] .- η_matrix[1,:]).^2)
    @show mean((lvs_matrix[111, :] .- η_matrix[1,:]).^2)
    @show mean((lvs_matrix[50, :] .- η_matrix[1,:]).^2)
    println()

    @show mean((lvs_matrix[92, :] .- η_matrix[2,:]).^2)
    @show mean((lvs_matrix[111, :] .- η_matrix[2,:]).^2)
    @show mean((lvs_matrix[50, :] .- η_matrix[2,:]).^2)
    println()

    @show mean((lvs_matrix[92, :] .- η_matrix[3,:]).^2)
    @show mean((lvs_matrix[111, :] .- η_matrix[3,:]).^2)
    @show mean((lvs_matrix[50, :] .- η_matrix[3,:]).^2)

    # Pair training and test data with their respective labels
    test_data = (lvs_matrix, η_matrix)

    test_dataloader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    test_dataloader = test_dataloader |> DEVICE

    model_name = "save_nr_$(load_model_nr).jld2"
    model_name_vae = "save_nr_$(load_model_nr_vae).jld2"


    synthetic_model = load("synthetic_saved_models/" * model_name, "model")
    vae = load("saved_models/" * model_name_vae, "vae")

    print_model(synthetic_model)

    synthetic_model.to_random_effects = synthetic_model.to_random_effects |> DEVICE
    synthetic_model = synthetic_model |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE # TODO test to remove this

    ps = params(synthetic_model)

    loss_saver_test = LossSaverSynthetic(0.0f0, 0.0f0)
    for (x_batch_test, y_batch_test) in test_dataloader
        images_test = vae.decoder(x_batch_test)

        loss(synthetic_model, images_test, y_batch_test, loss_saver_test)
    end

    loss_test = loss_saver_test.loss / loss_saver_test.counter
    println("Loss test:  $(Printf.@sprintf("%.9f", loss_test))")
end

main()
