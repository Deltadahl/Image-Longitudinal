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
    var_noise = 99.0
    epochs = 86
    load_model_nr = 0
    load_model_nr_vae = 526
    try_nr = 17

    evaluate_interval = 16000 * 6
    batch_size = 16

    # filepath = "data/synthetic/eta_and_lv_data_2_1M.jld2"
    filepath = "data/synthetic/noise_$(var_noise)_eta_approx_and_lv_data_1000k.jld2"
    η_matrix = load(filepath, "η_approx")
    # η_matrix = Float32.(hcat([t.η for t in η_approx]...))
    η_matrix = Float32.(η_matrix)
    lvs_matrix = load(filepath, "lvs_matrix")

    # Just to check that the MSE with the selected dim is much lower than with the other dims
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

    num_samples = size(lvs_matrix, 2)
    split_idx = Int(floor(0.95 * num_samples))
    # Split the data into training and test sets
    lvs_train = lvs_matrix[:, 1:split_idx]
    lvs_test = lvs_matrix[:, split_idx+1:end]
    η_train = η_matrix[:, 1:split_idx]
    η_test = η_matrix[:, split_idx+1:end]

    # Pair training and test data with their respective labels
    train_data = (lvs_train, η_train)
    test_data = (lvs_test, η_test)

    train_dataloader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_dataloader = DataLoader(test_data, batchsize=batch_size, shuffle=false)

    train_dataloader = train_dataloader |> DEVICE
    test_dataloader = test_dataloader |> DEVICE

    model_name = "save_nr_$(load_model_nr).jld2"
    model_name_vae = "save_nr_$(load_model_nr_vae).jld2"

    if load_model_nr > 0
        synthetic_model = load("synthetic_saved_models/" * model_name, "model")
    else
        to_random_effects = create_synthetic_model()
        synthetic_model = SyntheticModel(to_random_effects)
    end
    vae = load("saved_models/" * model_name_vae, "vae")

    print_model(synthetic_model)

    synthetic_model.to_random_effects = synthetic_model.to_random_effects |> DEVICE
    synthetic_model = synthetic_model |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE # Probably not needed.

    ps = params(synthetic_model)
    opt = ADAM(0.001)

    start_time = time()
    save_nr = 1
    if load_model_nr > 0
        save_nr += load_model_nr
    end

    for epoch in 1:epochs
        println("Epoch: $epoch/$epochs")
        loss_saver = LossSaverSynthetic(0.0f0, 0.0f0)
        for (batch_nr, (x_batch, y_batch)) in enumerate(train_dataloader)
            images = vae.decoder(x_batch)

            train!(synthetic_model, images, y_batch, opt, ps, loss_saver)

            if batch_nr * batch_size % evaluate_interval == 0
                loss_saver_test = LossSaverSynthetic(0.0f0, 0.0f0)
                for (x_batch_test, y_batch_test) in test_dataloader
                    images_test = vae.decoder(x_batch_test)

                    loss(synthetic_model, images_test, y_batch_test, loss_saver_test)
                end

                print_and_save(start_time, loss_saver, loss_saver_test, try_nr, save_nr)
                loss_saver = LossSaverSynthetic(0.0f0, 0.0f0)

                save_model(save_nr, synthetic_model)
                plot_losses(try_nr, evaluate_interval, split_idx)
                save_nr += 1
            end
        end
    end
    return nothing
end

main()
