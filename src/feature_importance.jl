using Images
using JLD2, FileIO
using Glob
using Flux
using Flux: params
using CUDA
using Statistics: mean
include("VAE.jl")
include("data_manipulation/data_loader_OCT.jl")

function get_feature_importance(vae::VAE)
    # Initialize the feature importance vector
    feature_importance = zeros(LATENT_DIM)
    n_images = 0

    for i in 1:200

        # μ = vae.μ_layer(vae.encoder(images))
        μ = randn(Float32, (LATENT_DIM, BATCH_SIZE)) |> DEVICE

        # Compute the perturbation
        perturbation = randn(Float32, size(μ)) |> DEVICE

        for i in 1:LATENT_DIM
            # Create a perturbed copy of the latent variables
            μ_perturbed = copy(μ)
            μ_perturbed[i, :] .= perturbation[i, :]

            # Decode the images from the original and perturbed latent variables
            decoded = vae.decoder(μ)
            decoded_perturbed = vae.decoder(μ_perturbed)

            # Compute the mean square error (MSE) between the decoded images
            mse = mean((decoded .- decoded_perturbed).^2, dims=(1,2,3))

            # Update the feature importance
            feature_importance[i] += sum(mse)
            n_images += size(mse, 1)  # Update the number of processed images
        end
    end

    # Normalize the feature importance vector
    feature_importance ./= n_images
    feature_importance = cpu(feature_importance)
    feature_importance_pairs = [(i, feature_importance[i]) for i in 1:LATENT_DIM]

    # Sort by importance, in descending order
    sorted_importance = sort(feature_importance_pairs, by = x -> x[2], rev=true)

    # Print features and their importance, from most to least important
    for (feature, importance) in sorted_importance
        println("Feature $feature, error $importance")
    end

    return feature_importance
end



function main()
    save_nr = 980

    model_path = "saved_models/save_nr_$(save_nr).jld2"
    vae = load(model_path, "vae")

    vae.decoder = vae.decoder |> DEVICE

    importance_scores = get_feature_importance(vae)
    @show importance_scores
    return nothing
end

@time main()

