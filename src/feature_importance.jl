using Images
using JLD2, FileIO
using Glob
using Flux
using Flux: params
using CUDA
using Statistics: mean
include("VAE.jl")
include("data_manipulation/data_loader_OCT.jl")


function get_feature_importance(vae::VAE, loader)
    # Initialize the feature importance vector
    feature_importance = zeros(LATENT_DIM)
    n_images = 0

    for (images, _) in loader
        if images === nothing
            break
        end
        images = images |> DEVICE

        # Compute the latent variables for the images
        encoded = vae.encoder(images)
        μ = vae.μ_layer(encoded)
        logvar = vae.logvar_layer(encoded)

        # Compute the perturbation
        perturbation = randn(Float32, size(μ)) |> DEVICE

        # Get the actual batch size
        actual_batch_size = size(μ, 2)

        for i in 1:LATENT_DIM
            for j in 1:actual_batch_size  # Iterate over the actual batch size
                # Create a perturbed copy of the latent variables
                μ_perturbed = copy(μ[:, j])
                μ_perturbed[i] += perturbation[i, j]

                # Decode the images from the original and perturbed latent variables
                decoded = vae.decoder(μ[:, j])
                decoded_perturbed = vae.decoder(μ_perturbed)

                # Compute the mean square error (MSE) between the decoded images
                mse = mean((decoded .- decoded_perturbed).^2)

                # Update the feature importance
                feature_importance[i] += mse
            end
        end

        # Keep track of the total number of images
        n_images += actual_batch_size  # Update the number of processed images
    end

    # Normalize the feature importance vector
    feature_importance ./= n_images

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
    importance_scores = get_feature_importance(vae, loader)
    @show importance_scores
    return nothing
end

main()