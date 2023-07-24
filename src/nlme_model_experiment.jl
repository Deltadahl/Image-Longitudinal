cd("/home/jrun/data/code/TemporalRetinaVAE/src")
using Pumas
using DeepPumas
using CairoMakie
using JLD2
using Images
using ThreadsX
include("constants.jl")
# include("VAE.jl")

pop_size = 1_000_000
η_size = 3
min_val = 1.0 * 10^(-8)
nlme_model = @model begin
  @param begin
    # tvImax ∈ RealDomain(; lower=min_val)
    # tvIC50 ∈ RealDomain(; lower=min_val)
    # tvKa ∈ RealDomain(; lower=min_val)
    # Ω ∈ PDiagDomain(3)
    # σ ∈ RealDomain(; lower=min_val)
    tvImax ∈ RealDomain(; lower=min_val, init=1.1)
    tvIC50 ∈ RealDomain(; lower=min_val, init=0.6)
    tvKa ∈ RealDomain(; lower=min_val)
    Ω ∈ PDiagDomain(; init = Diagonal([0.2, 0.2, 0.2]))
    σ ∈ RealDomain(; lower=min_val, init=0.1)
  end
  @random η ~ MvNormal(Ω)
  @pre begin
    Ka = tvKa * exp(η[1])
    Imax = tvImax * exp(η[2])
    IC50 = tvIC50 * exp(η[3])
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - Imax * Central / (IC50 + Central) # problem?
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end
# save_nr = 269
# model_path = "../saved_models/save_nr_$(save_nr).jld2"
# vae = load(model_path, "vae") # Use when creating images.

# selected_features = [25, 7, 1, 53, 87, 61, 45, 126, 29, 113, 3, 32, 6, 22, 44, 96, 81, 122, 56, 20, 103, 52, 65, 30, 121, 24, 35, 8, 47, 40, 120, 92, 27, 41, 71, 50, 34, 59, 106, 90, 18, 112, 5, 97, 37, 114, 98, 128, 36, 76, 38, 75, 69, 57, 28, 82, 99, 72, 23, 4, 116, 46, 17, 12, 43, 110, 63, 73, 85, 127, 108, 60, 102, 88, 70, 119, 39, 83, 115, 109, 107, 125, 14, 77, 51, 15, 64, 118, 31, 19, 68, 117, 80, 54, 79, 66, 67, 100, 21, 111, 105, 11, 74, 94, 78, 2, 123, 89, 10, 48, 42, 9, 58, 55, 49, 13, 16, 104, 95, 124, 93, 62, 84, 26, 33, 91, 86, 101]
selected_features = [92, 111, 50, 3, 91, 67, 37, 8, 90, 120, 54, 56, 21, 61, 75, 29, 80, 12, 95, 118, 73, 94, 101, 20, 48, 99, 104, 13, 59, 52, 106, 79, 4, 86, 93, 85, 72, 32, 87, 35, 47, 113, 40, 53, 36, 55, 122, 22, 5, 2, 88, 77, 26, 15, 7, 108, 58, 28, 39, 128, 126, 25, 103, 65, 105, 34, 18, 69, 27, 43, 64, 123, 38, 78, 17, 121, 42, 49, 33, 66, 57, 6, 24, 112, 10, 115, 68, 45, 11, 51, 41, 97, 70, 102, 114, 89, 71, 44, 110, 109, 62, 31, 124, 16, 1, 74, 9, 119, 14, 83, 117, 76, 60, 46, 23, 84, 98, 82, 100, 107, 81, 125, 127, 30, 19, 96, 63, 116]

nlme_params = (
		; tvImax=2.1,
		tvIC50=0.4,
		tvKa=1.0,
		# Ω=Diagonal([1.0, 1.0, 1.0]),
		Ω=Diagonal([0.2, 0.2, 0.2]),
		σ=0.01
	)

@time synth_data_pairs = map(1:pop_size) do i
    lv = randn(Float32, LATENT_DIM)
    η = (; η=lv[selected_features[1:3]])

    # img = vae.decoder(lv)
    img = "PLACE HOLDER"

    ## Create a subject that just stores some covariates and a dosing regimen
    no_obs_subj = Subject(;
        covariates=(; img, true_η=η.η, lv=lv), # Store some relevant info
        id=i,
        events=DosageRegimen(1.0)
    )

    ## Simulate some observations for the subject
    sim = simobs(
        nlme_model,
        no_obs_subj,
        nlme_params,
        η;
        obstimes=0:0.5:10 # whatever seems appropriate
    )

    ## Make a Subject of our simulation. The data from no_obs_subj will tag along.
    subj = Subject(sim)
    return (subj, img, lv, η)
end

pop = getindex.(synth_data_pairs, 1);
imgs = getindex.(synth_data_pairs, 2);
lvs = getindex.(synth_data_pairs, 3);
ηs = getindex.(synth_data_pairs, 4);

# idx = 74
# generated_image = vae.decoder(lvs[idx])
# Images.colorview(Gray, generated_image[:, :, 1, 1])

# plotgrid(pop[1:12])

####################### Back way #############################

@time fpm = fit(
  nlme_model,
    pop,
    init_params(nlme_model),
    FOCE()
)

# @time ebes = ThreadsX.map(pop) do subj
#   # @time ebes = map(pop) do subj
#   return empirical_bayes(
#     nlme_model,
#     subj,
#     nlme_params,
#     FOCE()
#   )
# end

@time η_pred = empirical_bayes(fpm)
ηs

# nlme_params
# pred = predict(fpm; obstimes=0:0.01:8)
# plotgrid(pred[1:8])

η_pred_matrix = hcat([t.η for t in η_pred]...)
# η_pred_matrix2 = hcat([t.η for t in ebes]...)
η_matrix = hcat([t.η for t in ηs]...)
error_fit = mean((η_pred_matrix .- η_matrix).^2)
# error_ebes = mean((η_pred_matrix2 .- η_matrix).^2)
error_random = mean((randn(size(η_matrix)) .- η_matrix).^2)
error_zero = mean((zeros(size(η_matrix)) .- η_matrix).^2)

# save("eta_and_img_data.jld2", "eta_matrix", η_pred_matrix, "imgs", imgs)
save("eta_and_lv_data.jld2", "eta_matrix", η_pred_matrix, "lvs", lvs)

stderror(infer(fpm))

# sim_orig = simobs(nlme_model, pop, nlme_params, ηs; obstimes=0:0.01:10, simulate_error=false)
# sim_approx = simobs(nlme_model, pop, nlme_params, η_pred; obstimes=0:0.01:10, simulate_error=false)
# plotgrid(sim_orig[1:12]; sim = (; markersize=0))
# plotgrid(sim_approx[1:12]; sim = (; markersize=0))
