using Images
using JLD2, FileIO
using Glob
using Flux
using Flux: params
using CUDA
using Statistics: mean
include("VAE.jl")
include("data_manipulation/data_loader_OCT.jl")

function get_feature_importance(vae::VAE, device)
    vae.encoder = vae.encoder |> device
    vae.μ_layer = vae.μ_layer |> device
    vae.logvar_layer = vae.logvar_layer |> device
    vae.decoder = vae.decoder |> device
    vae = vae |> device

    # Initialize the feature importance vector
    feature_importance = zeros(LATENT_DIM) |> device
    n_images = 0
    n_runs = 50000
    z = randn(Float32, (n_runs, LATENT_DIM)) |> device
    for i = 1:n_runs
        z_prim = vae.μ_layer(vae.encoder(vae.decoder(z[i, :])))

        mse = (z[i, :] .- z_prim).^2

        feature_importance .+= mse
    end

    # Normalize the feature importance vector
    feature_importance ./= n_runs
    feature_importance = cpu(feature_importance)
    feature_importance_pairs = [(i, feature_importance[i]) for i in 1:LATENT_DIM]

    # Sort by importance, in descending order
    sorted_importance = sort(feature_importance_pairs, by = x -> x[2], rev=false)

    # Print features and their importance, from most to least important
    for (feature, importance) in sorted_importance
        println("Feature $feature, error $importance")
    end

    return feature_importance
end



function main()
    save_nr = 289

    model_path = "saved_models/save_nr_$(save_nr).jld2"
    vae = load(model_path, "vae")

    device = DEVICE
    importance_scores = get_feature_importance(vae, device)
    @show importance_scores
    return nothing
end

@time main()
