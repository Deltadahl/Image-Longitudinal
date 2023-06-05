using Flux
using Flux: @epochs, mse, params, throttle
using Flux: Conv, Dense, MaxPool, BatchNorm, flatten
using Flux: Chain, throttle, @functor
using Images
using Statistics: mean
using FileIO
using ImageMagick
using ImageTransformations
using Images
using IterTools: partition
using Shuffle: shuffle
using BSON: @save, @load
using CUDA

const BATCH_SIZE = 64
const EPOCHS = 1 # TODO add more when I got a working program

# Define the encoder
function create_encoder()
    return Chain(
        Conv((3, 3), 1 => 32, stride = 2, pad = SamePad(), relu),
        Conv((3, 3), 32 => 64, stride = 2, pad = SamePad(), relu),
        Conv((3, 3), 64 => 128, stride = 2, pad = SamePad(), relu),
        flatten,
        Dense(8192, 256, relu), # TODO might just ignore this layer?
    )
end

# Define the mean and log variance layers
function create_mu_logvar_layers()
    return Dense(256, 64), Dense(256, 64)
end

# Define the decoder
function create_decoder()
    return Chain(
        Dense(64, 8192, relu),
        x -> reshape(x, (8, 8, 128, :)),
        ConvTranspose((3, 3), 128 => 64, stride = 2, pad = SamePad(), relu),
        ConvTranspose((3, 3), 64 => 32, stride = 2, pad = SamePad(), relu),
        ConvTranspose((3, 3), 32 => 1, stride = 2, pad = SamePad(), sigmoid),
    )
end

# Define the VAE
struct VAE
    encoder::Any
    mu_layer::Any
    logvar_layer::Any
    decoder::Any
end

@functor VAE

function (m::VAE)(x)
    encoded = m.encoder(x)
    mu = m.mu_layer(encoded)
    logvar = m.logvar_layer(encoded)
    # z = mu + exp.(0.5 .* logvar) #.* CUDA.randn(Float32, size(mu)) # TODO uncomment
    r = randn(Float32, size(mu)) |> gpu # TODO see if there is a better workaround and double check that no differentiation is happening
    z = mu + exp.(0.5 .* logvar) .* r
    decoded = m.decoder(z)
    return decoded, mu, logvar
end

# Define the loss function
function loss(x, m::VAE)
    decoded, mu, logvar = m(x)
    reconstruction_loss = mse(decoded, x)
    kl_divergence = -0.5 .* sum(1 .+ logvar .- mu .^ 2 .- exp.(logvar))
    return reconstruction_loss + kl_divergence
end

# function load_and_preprocess_image(img_path)
#     img = load(img_path) # Load image
#     img = Gray.(img) # Convert to grayscale
#     img = imresize(img, (64, 64)) # Resize to 64x64 # TODO change
#     return Float32.(channelview(img)) # Convert to Float32 array
# end

# function load_images_from_dir(dir_path, num_images=Inf) # num_images defaults to Inf, i.e. all images
#     img_files = readdir(dir_path) # Get all files in directory
#     img_files = filter(x -> occursin(".jpeg", x), img_files) # Filter to only .jpeg files
#     img_files = img_files[1:min(end, num_images)] # Take only the first num_images files
#     # Load and preprocess images in chunks
#     chunk_size = 1000  # Set a reasonable chunk size
#     num_chunks = ceil(Int, length(img_files) / chunk_size)
#     all_imgs = []
#     for i in 1:num_chunks
#         start_idx = (i-1)*chunk_size + 1
#         end_idx = min(i*chunk_size, length(img_files))
#         imgs = [load_and_preprocess_image(joinpath(dir_path, img_file)) for img_file in img_files[start_idx:end_idx]]
#         push!(all_imgs, cat(imgs..., dims=4)) # Concatenate along 4th dimension
#     end
#     return cat(all_imgs..., dims=4)  # Concatenate all chunks along 4th dimension
# end

# function load_all_data(data_dirs, num_images_per_dir=Inf) # num_images_per_dir defaults to Inf
#     data = []
#     for dir in data_dirs
#         push!(data, load_images_from_dir(dir, num_images_per_dir))
#     end
#     return cat(data..., dims=4) # Concatenate along 4th dimension
# end


# TODO is there a better way to do this?, maybe mine is the same order every time?
function create_minibatches(data, batch_size)
    n = size(data, 4)
    shuffled_indices = shuffle(1:n) # Shuffle indices
    data = data[:, :, :, shuffled_indices] # Shuffle data
    return [data[:, :, :, collect(inds)] for inds in partition(1:n, batch_size)] # Create minibatches
end

function main()
    # Initialize the VAE
    encoder = create_encoder()
    mu_layer, logvar_layer = create_mu_logvar_layers()
    decoder = create_decoder()
    vae = VAE(encoder, mu_layer, logvar_layer, decoder)

    # Move the model to the GPU
    vae = gpu(vae)

    # Define the optimizer
    opt = ADAM()

    @load "data/all_data.bson" data
    minibatches = create_minibatches(data, BATCH_SIZE)

    # Move the data to the GPU
    minibatches = gpu.(minibatches)

    println("Done loading images.")

    # Training the model
    println("Training the model...")
    # TODO use? @epochs EPOCHS Flux.train!(loss, ps, data, opt)
    ps = Flux.params(vae)
    for epoch = 1:EPOCHS
        for minibatch in minibatches
            gs = Flux.gradient(ps) do
                l = loss(minibatch, vae)
                return l
            end
            Flux.update!(opt, ps, gs)
        end
    end
    println("Done training the model.")

    # Save the model
    @save "data/saved_models/vae_develop.bson" vae
end

@time main()
