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
    load_model_nr = 45
    load_model_nr_vae = 526
    try_nr = 8

    batch_size = 24

    filepath = "data/synthetic/eta_approx_and_lv_data_100k.jld2"
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
    imgs_to_consider = 100_000
    η_approx = η_matrix[:,1:imgs_to_consider]
    test_data = (lvs_matrix[:,1:imgs_to_consider], η_approx) # TODO use all in final version

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

    η_pred = zeros(Float32, 3, imgs_to_consider)
    imgs_list = zeros(Float32, 224, 224, 1, imgs_to_consider)
    for (idx, (x_batch_test, y_batch_test)) in enumerate(test_dataloader)

        imgs = vae.decoder(x_batch_test)
        y_approx = synthetic_model(imgs)
        y_approx = y_approx |> cpu
        imgs = imgs |> cpu
        η_pred[:, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)] = y_approx
        imgs_list[:, :, :, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)] = imgs
    end


    η_true = lvs_matrix[[92, 111, 50], 1:imgs_to_consider]
    @show mean((η_pred .- η_approx).^2)
    @show mean((η_pred .- η_true).^2)
    @show mean((η_approx .- η_true).^2)
    @show mean((η_pred .- zeros(size(η_pred))).^2)
    @show mean((η_pred .- randn(size(η_pred))).^2)
    @show mean((η_approx .- zeros(size(η_pred))).^2)
    @show mean((η_approx .- randn(size(η_pred))).^2)
    @show mean((η_true .- zeros(size(η_pred))).^2)
    @show mean((η_true .- randn(size(η_pred))).^2)
    @show mean(η_pred)
    @show var(η_pred)
    @show mean(η_approx)
    @show var(η_approx)
    @show mean(η_true)
    @show var(η_true)

    @show size(η_pred)

    # for i = 1:20
    #     @show η_pred[:,i]
    #     @show η_matrix[:,i]
    #     @show η_true[:,i]
    #     println()
    # end

    # filename = "data/synthetic/eta_pred.jld2"
    # save(filename, "eta_pred", η_pred)

    # for i = 1:imgs_to_consider
    #     img = Images.colorview(Gray, imgs_list[:,:,1,i])
    #     save("data/synthetic/imgs/img_$(i).png", img)
    # end



    # Function to calculate MSE on bootstrap sample
    mse(x::Matrix) = mean((x[:,1] .- x[:,2]).^2)
    n_bootstraps = 10000
    Random.seed!(1)
    function get_bootstrap(η_1, η_2)
        η_matrix = hcat(vec(η_1), vec(η_2))
        bootstrap_results = bootstrap(mse, η_matrix, BasicSampling(n_bootstraps))
        bootstrap_ci = confint(bootstrap_results, BasicConfInt(0.95))
    end
    @show get_bootstrap(η_pred, η_approx)
    @show get_bootstrap(η_pred, η_true)
    @show get_bootstrap(η_approx, η_true)
    @show get_bootstrap(η_pred, zeros(size(η_pred)))
    @show get_bootstrap(η_pred, randn(size(η_pred)))
    @show get_bootstrap(η_approx, zeros(size(η_pred)))
    @show get_bootstrap(η_approx, randn(size(η_pred)))
    @show get_bootstrap(η_true, zeros(size(η_pred)))
    @show get_bootstrap(η_true, randn(size(η_pred)))



    return nothing
end

@time main()




# mean((η_predicted .- η_approx) .^ 2) = 0.06284952f0
# mean((η_predicted .- η_true) .^ 2) = 0.24383068f0
# mean((η_approx .- η_true) .^ 2) = 0.30281532f0
# mean((η_predicted .- zeros(size(η_predicted))) .^ 2) = 0.6249630370084249
# mean((η_predicted .- randn(size(η_predicted))) .^ 2) = 1.6274439311071005
# mean((η_approx .- zeros(size(η_predicted))) .^ 2) = 0.6760396509994607
# mean((η_approx .- randn(size(η_predicted))) .^ 2) = 1.6780876723708422
# mean((η_true .- zeros(size(η_predicted))) .^ 2) = 0.9963293177459422
# mean((η_true .- randn(size(η_predicted))) .^ 2) = 1.9936094051722197
# mean(η_predicted) = -0.05673421f0
# var(η_predicted) = 0.62174636f0
# mean(η_approx) = -0.051700324f0
# var(η_approx) = 0.673369f0
# mean(η_true) = -0.0010691533f0
# var(η_true) = 0.99633145f0
