# synthetic_train_model.jl
using JLD2, FileIO
using Flux
using Flux.Optimise
using Flux: params
using Glob
using Printf
using CUDA
using Flux.Data: DataLoader
using Images
using Bootstrap
using Random
using Statistics
include("synthetic_model.jl")
include("synthetic_utils_train.jl")
include("data_manipulation/plot_losses_synthetic.jl")
include("VAE.jl")

function main()
    var_noise = 0.0
    # load_model_nr = 86
    load_model_nr = 36
    load_model_nr_vae = 526
    batch_size = 12
    imgs_to_consider = 100_000
    # imgs_to_consider = 1000

    image_data_dir = "saved_eta_and_lv_data/json/output_test/" #TODO ....#######.....
    filepath = "saved_eta_and_lv_data/noise_$(var_noise)_eta_approx_and_lv_data_100k.jld2"
    # filepath = "data/synthetic/noise_$(var_noise)_eta_approx_and_lv_data_100k.jld2"
    η_matrix = load(filepath, "η_approx")
    η_matrix = Float32.(η_matrix)
    lvs_matrix = load(filepath, "lvs_matrix")
    # η_true = load(filepath, "η_true_noise")
    η_true = lvs_matrix[[92, 111, 50], 1:imgs_to_consider]

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

    # Pair training and test data with their respective labels
    η_approx = η_matrix[:,1:imgs_to_consider]
    test_data = (collect(1:imgs_to_consider), η_approx)
    # test_data = (lvs_matrix[:,1:imgs_to_consider], η_approx)

    test_dataloader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    test_dataloader = test_dataloader |> DEVICE

    model_name = "noise_$(var_noise)_save_nr_$(load_model_nr).jld2"

    synthetic_model = load("synthetic_saved_models/" * model_name, "model")
    # vae = load("saved_models/" * model_name_vae, "vae")

    print_model(synthetic_model)

    synthetic_model.to_random_effects = synthetic_model.to_random_effects |> DEVICE
    synthetic_model = synthetic_model |> DEVICE
    # vae.decoder = vae.decoder |> DEVICE
    # vae = vae |> DEVICE # Can probobly be removed.

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

    η_pred = zeros(Float32, 3, imgs_to_consider)
    # imgs_list = zeros(Float32, 224, 224, 1, imgs_to_consider)
    for (idx, (x_batch_test, y_batch_test)) in enumerate(test_dataloader)
        if idx % 50 == 0
            println("idx: ", idx)
        end

        batch_images = [CUDA.cu(Float32.(preprocess_image(images[x]))) for x in Array(x_batch_test)]
        stacked_images = cat(batch_images..., dims=4)
        imgs = reshape(stacked_images, size(stacked_images, 1), size(stacked_images, 2), 1, size(stacked_images, 4))

        ######
        cpu_y_batch_test = y_batch_test |> cpu
        if abs(cpu_y_batch_test[3,1]) > 2.0
            cpu_img_tmp = imgs |> cpu
            println("saving image ...")
            save("TEST/$(cpu_y_batch_test[3,1]).jpeg", cpu_img_tmp[:, :, 1, 1])
        end
        ######

        # imgs = vae.decoder(x_batch_test)
        y_approx = synthetic_model(imgs)
        y_approx = y_approx |> cpu
        # imgs = imgs |> cpu
        η_pred[:, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)] = y_approx
        # imgs_list[:, :, :, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)] = imgs
        # @show y_approx
        # @show y_batch_test
        # @show η_true[:, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)]
        # println()
    end


    @show mean((η_pred .- η_approx).^2)
    @show mean((η_pred .- η_true).^2)
    @show mean((η_approx .- η_true).^2)
    @show mean((η_pred .- zeros(size(η_pred))).^2)
    # @show mean((η_pred .- randn(size(η_pred))).^2)
    @show mean((η_approx .- zeros(size(η_pred))).^2)
    # @show mean((η_approx .- randn(size(η_pred))).^2)
    @show mean((η_true .- zeros(size(η_pred))).^2)
    # @show mean((η_true .- randn(size(η_pred))).^2)
    @show mean(η_pred)
    @show var(η_pred)
    @show mean(η_approx)
    @show var(η_approx)
    @show mean(η_true)
    @show var(η_true)
    # @show size(η_pred)

    # filename = "data/synthetic/noise_$(var_noise)_eta_pred.jld2"
    # save(filename, "η_pred", η_pred)

    # # Function to calculate MSE on bootstrap sample
    # mse(x::Matrix) = mean((x[:,1] .- x[:,2]).^2)
    # n_bootstraps = 1000 # 10000
    # function get_bootstrap(η_1, η_2)
    #     η_matrix = hcat(vec(η_1), vec(η_2))
    #     bootstrap_results = bootstrap(mse, η_matrix, BasicSampling(n_bootstraps))
    #     bootstrap_ci = confint(bootstrap_results, BasicConfInt(0.95))
    # end
    # @show get_bootstrap(η_pred, η_approx)
    # @show get_bootstrap(η_pred, η_true)
    # @show get_bootstrap(η_approx, η_true)
    # @show get_bootstrap(η_pred, zeros(size(η_pred)))
    # @show get_bootstrap(η_pred, randn(size(η_pred)))
    # @show get_bootstrap(η_approx, zeros(size(η_pred)))
    # @show get_bootstrap(η_approx, randn(size(η_pred)))
    # @show get_bootstrap(η_true, zeros(size(η_pred)))
    # @show get_bootstrap(η_true, randn(size(η_pred)))

    return nothing
end

@time main()
