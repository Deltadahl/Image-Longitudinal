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
include("synthetic_model.jl")
include("synthetic_utils_train.jl")
include("data_manipulation/plot_losses_synthetic.jl")
include("VAE.jl")

function main()
    load_model_nr = 58
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
    test_data = (lvs_matrix[:,1:imgs_to_consider], η_matrix[:,1:imgs_to_consider]) # TODO use all in final version

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

    η_approx = zeros(Float32, 3, imgs_to_consider)
    imgs_list = zeros(Float32, 224, 224, 1, imgs_to_consider)
    for (idx, (x_batch_test, y_batch_test)) in enumerate(test_dataloader)

        imgs = vae.decoder(x_batch_test)
        y_approx = synthetic_model(imgs)
        y_approx = y_approx |> cpu
        imgs = imgs |> cpu
        η_approx[:, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)] = y_approx
        imgs_list[:, :, :, (idx-1)*batch_size+1:min(idx*batch_size, imgs_to_consider)] = imgs
    end

    @show mean((η_approx .- η_matrix[:,1:imgs_to_consider]).^2)
    lvs_matrix_92_111_50 = lvs_matrix[[92, 111, 50], 1:imgs_to_consider]
    @show mean((η_approx .- lvs_matrix_92_111_50).^2)
    @show mean((η_matrix[:,1:imgs_to_consider] .- lvs_matrix_92_111_50).^2)

    # for i = 1:20
    #     @show η_approx[:,i]
    #     @show η_matrix[:,i]
    #     @show lvs_matrix_92_111_50[:,i]
    #     println()
    # end

    # filename = "data/synthetic/eta_approx.jld2"
    # save(filename, "eta_approx", η_approx)

    # for i = 1:imgs_to_consider
    #     img = Images.colorview(Gray, imgs_list[:,:,1,i])
    #     save("data/synthetic/imgs/img_$(i).png", img)
    # end

    return nothing
end

@time main()
