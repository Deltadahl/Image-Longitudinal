# get_feature_matrix.jl
# using Images
using JLD2, FileIO
using Glob
using Flux
using CUDA
using LinearAlgebra
include("data_manipulation/data_loader_OCT.jl")
include("VAE.jl")


function get_matrix(vae, loader; epoch=0)
    # excluded_features = [7, 92, 78]
    pop_size = 100

    η_vae = zeros(Float32, LATENT_DIM, pop_size)
    num_added = 0  # Keep track of the number of added samples

    for (images, _) in loader
        if images === nothing || num_added >= pop_size
            break
        end
        images = images |> DEVICE
        μ = vae.μ_layer(vae.encoder(images))

        batch_size = min(size(μ, 2), pop_size - num_added)  # Add up to 100 samples
        η_vae[:, num_added+1:num_added+batch_size] = μ[:, 1:batch_size]
        num_added += batch_size
    end

    # Save the result matrix
    @show η_vae[:, 1]
    @save "output_matrix/testing_output.jld2" η_vae

    return nothing
end

function main()
    data_name = "OCT"
    data_path = "data/data_resized/bm3d_224_train_100"
    epoch = 9

    model_path = "saved_models/$(data_name)_epoch_$(epoch).jld2"
    vae = load(model_path, "vae")

    vae.encoder = vae.encoder |> DEVICE
    vae.μ_layer = vae.μ_layer |> DEVICE
    vae.logvar_layer = vae.logvar_layer |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE

    loader = DataLoaderOCT(data_path, BATCH_SIZE, false)
    get_matrix(vae, loader)
    return nothing
end

main()
