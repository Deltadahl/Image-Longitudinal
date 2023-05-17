using Flux
include("VAE.jl")

# Create the VAE model
encoder = create_encoder()
mu_layer, logvar_layer = create_mu_logvar_layers()
decoder = create_decoder()
model = VAE(encoder, mu_layer, logvar_layer, decoder)
@functor VAE

data_loader = open(deserialize, "dataloader.jls")
