# testing_2.jl
using Flux
using Flux.Optimise
using Flux: params
using FileIO
using Glob
using CUDA
using JLD2
CUDA.reclaim()
include("VAE_test.jl")
include("data_manipulation/data_loader_MNIST_test.jl")

function main()
    data_name = "MNIST"
    data_path = "data/MNIST_small"
    save_model = false
    load_model = true
    save_path = "saved_models/dev.jld2"
    loader = DataLoaderMNIST(data_path, BATCH_SIZE)


    # Initialize CUDA before loading the model
    # CUDA.allowscalar(false)
    # CUDA.device!(0)  # change 0 to the ID of your GPU, if you have more than one

    if load_model
        vae = load(save_path, "vae")
    else
        encoder = create_encoder()
        μ_layer, logvar_layer = create_μ_logvar_layers()
        decoder = create_decoder()
        vae = VAE(encoder, μ_layer, logvar_layer, decoder)
    end
    if save_model
        save(save_path, "vae", vae)
    end
    vae.encoder = vae.encoder |> DEVICE
    vae.μ_layer = vae.μ_layer |> DEVICE
    vae.logvar_layer = vae.logvar_layer |> DEVICE
    vae.decoder = vae.decoder |> DEVICE
    vae = vae |> DEVICE
    println(vae)

    println("test 1")
    for (images, labels) in loader
        if images === nothing
            break
        end
        images = images |> DEVICE
        labels = labels |> DEVICE
        output = vae(images)
        # show size of output
        @show length(output)
        break
    end

    return nothing
end

main()
