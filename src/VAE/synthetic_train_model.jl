using Pkg
Pkg.activate("/store/CIA/scfc3/diffusion/TemporalRetinaVAE")
cd("/store/CIA/scfc3/diffusion/TemporalRetinaVAE")
# synthetic_train_model.jl
using Printf
using Base: match
using JLD2, FileIO
using Flux
using Flux.Optimise
using Flux: params
using Glob
using Printf
using CUDA
using Flux.Data: DataLoader
using CSV
using LinearAlgebra
using ImageTransformations
include("synthetic_model.jl")
include("synthetic_utils_train.jl")
include("data_manipulation/plot_losses_synthetic.jl")
# include("VAE.jl")

function main()
    # var_noise = 49.0  # NOTE
    epochs = 100
    load_model_nr = 0
    # load_model_nr_vae = 526
    try_nr = 111   # NOTE

    batch_size = 16
    evaluate_interval = 10000 * batch_size

    image_data_dir = "saved_eta_and_lv_data/json/output/"
    # filepath = "saved_eta_and_lv_data/noise_$(var_noise)_eta_approx_and_lv_data_1000k.jld2"
    filepath = "saved_eta_and_lv_data/NEW_DATA/noise_1.0_eta_approx_and_lv_data_1000k.jld2"   # NOTE
    η_matrix = load(filepath, "η_approx")
    η_matrix = Float32.(η_matrix)
    lvs_matrix = load(filepath, "lvs_matrix")

    # Just to check that the MSE with the selected dim is much lower than with the other dims
    # mse lvs_matrix 92, 111 and 50 vs η_matrix
    @show mean((lvs_matrix[92, :] .- η_matrix[1, :]) .^ 2)
    @show mean((lvs_matrix[111, :] .- η_matrix[1, :]) .^ 2)
    @show mean((lvs_matrix[50, :] .- η_matrix[1, :]) .^ 2)
    println()
    @show mean((lvs_matrix[92, :] .- η_matrix[2, :]) .^ 2)
    @show mean((lvs_matrix[111, :] .- η_matrix[2, :]) .^ 2)
    @show mean((lvs_matrix[50, :] .- η_matrix[2, :]) .^ 2)
    println()
    @show mean((lvs_matrix[92, :] .- η_matrix[3, :]) .^ 2)
    @show mean((lvs_matrix[111, :] .- η_matrix[3, :]) .^ 2)
    @show mean((lvs_matrix[50, :] .- η_matrix[3, :]) .^ 2)

    num_samples = size(lvs_matrix, 2)
    split_idx = Int(floor(0.95 * num_samples))
    # Split the data into training and test sets
    lvs_train = lvs_matrix[:, 1:split_idx]
    lvs_test = lvs_matrix[:, split_idx+1:end]
    η_train = η_matrix[:, 1:split_idx]
    η_test = η_matrix[:, split_idx+1:end]

    # Pair training and test data with their respective labels
    train_data = (collect(1:split_idx), η_train)
    test_data = (collect(split_idx+1:num_samples), η_test)

    train_dataloader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_dataloader = DataLoader(test_data, batchsize=batch_size, shuffle=false)

    train_dataloader = train_dataloader |> DEVICE
    test_dataloader = test_dataloader |> DEVICE

    model_name = "save_nr_$(load_model_nr).jld2"
    # model_name_vae = "save_nr_$(load_model_nr_vae).jld2"

    if load_model_nr > 0
        synthetic_model = load("synthetic_saved_models/" * model_name, "model")
    else
        to_random_effects = create_synthetic_model()
        synthetic_model = SyntheticModel(to_random_effects)
    end
    # vae = load("saved_models/" * model_name_vae, "vae")

    print_model(synthetic_model)

    synthetic_model.to_random_effects = synthetic_model.to_random_effects |> DEVICE
    synthetic_model = synthetic_model |> DEVICE
    # vae.decoder = vae.decoder |> DEVICE
    # vae = vae |> DEVICE # Probably not needed.

    ps = params(synthetic_model)
    opt = ADAM(0.001)

    start_time = time()
    save_nr = 1
    if load_model_nr > 0
        save_nr += load_model_nr
    end


    # Helper function to extract the numerical part of the filename
    function extract_number(filename::String)
        # Attempt to match the numerical part of the filename
        match_result = match(r"image_(\d+)\.jpeg", filename)
        if match_result !== nothing
            # If a match is found, parse the number and return it
            return parse(Int, match_result.captures[1])
        else
            # If no match is found, return a default value
            return -1
        end
    end

    # Adjusted function to collect and then sort all images across subfolders
    function load_images_and_sort_globally(image_subfolders)
        images = []
        for subfolder in image_subfolders
            image_files = readdir(subfolder, join=true)
            append!(images, image_files)
        end
        # Sort all collected image paths globally based on the extracted number
        sort!(images, by=filename -> extract_number(String(split(filename, '/')[end])))
        images
    end

    image_subfolders = readdir(image_data_dir, join=true)

    images = load_images_and_sort_globally(image_subfolders) # Ensure this function loads images or paths

    function preprocess_image(path)
        img = load(path) # Load the image from path
        img_resized = imresize(img, (224, 224)) # Resize the image
        return img_resized
    end

    # Optional: Preprocess images if not already done
    # images = [preprocess_image(img_path) for img_path in images]
    best_val_loss = Inf

    for epoch in 1:epochs
        println("Epoch: $epoch/$epochs")
        loss_saver = LossSaverSynthetic(0.0f0, 0.0f0)
        for (batch_nr, (x_batch, y_batch)) in enumerate(train_dataloader)
            if batch_nr % 100 == 0
                println("Batch: $batch_nr")
            end

            batch_images = [CUDA.cu(Float32.(preprocess_image(images[x]))) for x in Array(x_batch)]
            # Stack images into a 4D array and add channel dimension
            stacked_images = cat(batch_images..., dims=4)
            stacked_images = reshape(stacked_images, size(stacked_images, 1), size(stacked_images, 2), 1, size(stacked_images, 4))

            # ######
            # cpu_y_batch_test = y_batch |> cpu
            # if abs(cpu_y_batch_test[3, 1]) > 2.0
            #     cpu_img_tmp = stacked_images |> cpu
            #     println("saving image ...")
            #     save("TEST/$(cpu_y_batch_test[3,1]).jpeg", cpu_img_tmp[:, :, 1, 1])
            # end
            # ######

            train!(synthetic_model, stacked_images, y_batch, opt, ps, loss_saver)

            if batch_nr * batch_size % evaluate_interval == 0
                println(save_nr)
                loss_saver_test = LossSaverSynthetic(0.0f0, 0.0f0)
                for (x_batch_test, y_batch_test) in test_dataloader
                    batch_images = [Float32.(preprocess_image(images[x])) for x in Array(x_batch_test)] |> DEVICE
                    stacked_images = cat(batch_images..., dims=4)
                    stacked_images = reshape(stacked_images, size(stacked_images, 1), size(stacked_images, 2), 1, size(stacked_images, 4))

                    # images_test = vae.decoder(x_batch_test)
                    loss(synthetic_model, stacked_images, y_batch_test, loss_saver_test)

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
