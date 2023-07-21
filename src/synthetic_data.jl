using JLD2, FileIO
using Flux
using Flux.Optimise
using Flux: params
using Glob
using Printf
using CUDA
include("VAE.jl")
include("generate_image.jl")
include("utils_train.jl")
include("data_manipulation/plot_losses.jl")


function main()
    # Load data
    filepath = "data/synthetic/eta_and_lv_data.jld2"
    η_matrix = load(filepath, "eta_matrix")
    lvs = load(filepath, "lvs")

    filepath = "data/synthetic/eta_and_lv_data.jld2"
    η_matrix = load(filepath, "eta_matrix")
    lvs = load(filepath, "lvs")
    epochs = 100000
    load_model_nr = 0
    try_nr = 1
    # evaluate_interval = 5000
    evaluate_interval = 5000

    model_name = "save_nr_$(load_model_nr).jld2"

    if load_model_nr > 0
        vae = load("saved_models/" * model_name, "vae")
    else
        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder)
    end

    # To print the VAE structure:
    print_vae(vae)
    vae_to_device!(vae, DEVICE)

    ps = params(vae)

    # step_decay = StepDecay(0.001, 0.9, 1)  # create a scheduler
    opt = ADAM(0.001)  # initialize with initial learning rate

    vgg = vgg_subnetworks()

    start_time = time()
    save_nr = 1
    if load_model_nr > 0
        save_nr += load_model_nr
    end

    for epoch in 1:epochs
        println("Epoch: $epoch/$epochs")
        loss_normalizers, loss_saver = reset_normalizers()
        loss_normalizers_test, loss_saver_test = reset_normalizers()
        statistics_saver = StatisticsSaver()

        loader = get_dataloader(data_name, data_path, BATCH_SIZE, true)
        for (batch_nr, (images, _)) in enumerate(loader)
            if images === nothing
                break
            end

            images = images |> DEVICE
            β_nr = (batch_nr * BATCH_SIZE + (epoch - 1) * IMAGES_TRAIN + load_model_nr * evaluate_interval) / IMAGES_TRAIN
            train!(vae, images, opt, ps, loss_saver, vgg, loss_normalizers, β_nr, statistics_saver, true)

            if batch_nr * BATCH_SIZE % evaluate_interval == 0
                loader_test = get_dataloader(data_name, data_path_test, BATCH_SIZE, false)
                for (images_test, _) in loader_test
                    if images_test === nothing
                        break
                    end
                    images_test = images_test |> DEVICE

                    loss(vae, images_test, loss_saver_test, vgg, loss_normalizers_test, β_nr, statistics_saver, false)
                end
                loader_eval = get_dataloader(data_name, data_path, BATCH_SIZE, false)
                output_image(vae, loader_eval; epoch=save_nr)
                print_and_save(start_time, loss_saver, loss_normalizers, loss_saver_test, loss_normalizers_test, try_nr, save_nr)
                print_statistics(statistics_saver, try_nr)
                statistics_saver = StatisticsSaver()
                loss_normalizers, loss_saver = reset_normalizers()
                loss_normalizers_test, loss_saver_test = reset_normalizers()
                save_model(save_nr, vae)
                plot_losses(try_nr, evaluate_interval)
                save_nr += 1

            end
        end
    end
    return nothing
end

main()
