# Master's Thesis: Bridging Medical Images and Longitudinal Data: Synthetic Dataset Generation via VAEs and NLME Models

This repository holds the code for the Master's Thesis titled "Bridging Medical Images and Longitudinal Data: Synthetic Dataset Generation via VAEs and NLME Models" by Simon Carlson. The research was conducted at the Department of Mathematical Sciences, Chalmers University of Technology, Sweden, in collaboration with Pumas-AI, Inc., USA. The thesis was defended on August 24, 2023.

## For Pumas-AI Team

### Trained Variational Autoencoder (VAE)

1. **Location**: The trained VAE is saved in `saved_models/save_nr_526.jld2`.
    - **Loading the Model**: Execute `vae = load("saved_models/save_nr_526.jld2", "vae")`. This is demonstrated in `src/train_model.jl`. Ensure the necessary files, like `src/VAE.jl`, are included.
    - **GPU Acceleration**: To move the model to the GPU for faster computation, run `vae_to_device!(vae, gpu)`.
    - **Usage Example**:
        ```julia
        z = randn(Float32, LATENT_DIM) |> DEVICE
        generated_img = vae.decoder(z)
        ```
    - **Architecture**: Refer to `src/VAE.jl` for the architecture.
    - **Training Script**: The model is trained using `src/train_model.jl`.

### OCT Image Data

2. **Raw Data**: Found in `data/CellData/OCT/`.
    - **Preprocessed Data**: Resized to 224x224 and located at `data/data_resized`.

### Synthetic Data

3. - **Location**: `data/synthetic/`.
    - **Synthetic Images**: The folder `imgs_100k` contains 100,000 synthetic images. The same images are used for all noise levels. Additional images can be easily generated with the VAE's decoder (see 1.3).
    - **Latent Variables and Random Effects**: Files named `noise_NOISE_eta_approx_and_lv_data_100k.jld2` contain various data. The `NOISE` variable can be {0.0, 1.0, 9.0, 18.0, 49.0}.
    - **File Contents**: The data is stored in a dictionary with keys `"lvs_matrix", "η_approx", and "η_true_noise"`.
    - **Longitudinal Data**: Generated in `src/nlme_model_use_data.jl`.

### Neural Networks (NNs)

4. - **Architecture**: Found in `src/synthetic_model.jl`.
    - **Training Script**: `src/train_synthetic_model.jl`.
    - **Usage**: `src/synthetic_model_use_data.jl`.
    - **Trained Models**: Saved in `synthetic_saved_models/noise_NOISE_save_nr_86.jld2`.

---

## Compatibility Issues Note

Due to compatibility issues between Metalhead and MLJFlux, I was unable to use the latest versions of Pumas and DeepPumas alongside my VAE and NN models. Consequently, I used two separate environments:
1. The primary `Manifest.toml` and `Project.toml` files in this project were used for all files except those beginning with `src/nlme_model...`.
2. For the six files in the `src` folder that start with `nlme_model`, the "JuliaHubEnv" `.toml` files were used.
