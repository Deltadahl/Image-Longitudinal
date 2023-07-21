cd("/home/jrun/data/code/TemporalRetinaVAE/src")
using Pumas
using DeepPumas
using CairoMakie
using JLD2
using Images
include("constants.jl")
include("VAE.jl")

pop_size = 100
η_size = 3

nlme_model = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0.)
    tvIC50 ∈ RealDomain(; lower=0.)
    tvKa ∈ RealDomain(; lower=0.)
    Ω ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(Ω)
  @pre begin
    Ka = tvKa * exp(η[1])
    Imax = tvImax * exp(η[2])
    IC50 = tvIC50 * exp(η[3])
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - Imax * Central / (IC50 + Central)
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end
save_nr = 289
model_path = "../saved_models/save_nr_$(save_nr).jld2"
vae = load(model_path, "vae")

# selected_features = [75, 25, 98, 18, 107, 101, 33, 60, 52, 34, 47, 116, 102, 5, 113, 43, 128, 14, 41, 16, 38, 12, 4, 115, 105, 26, 64, 32, 1, 103, 59, 63, 21, 56, 31, 72, 46, 36, 108, 104, 89, 82, 124, 97, 122, 53, 84, 55, 57, 65, 6, 94, 39, 15, 78, 74, 49, 50, 96, 7, 83, 81, 93, 17, 20, 69, 87, 125, 112, 28, 90, 110, 40, 117, 2, 99, 10, 88, 118, 29, 30, 42, 23, 119, 67, 19, 44, 126, 70, 73, 45, 114, 35, 106, 86, 109, 85, 77, 71, 62, 54, 51, 13, 91, 123, 66, 27, 58, 61, 68, 11, 37, 76, 48, 120, 8, 3, 80, 79, 121, 100, 92, 24, 9, 111, 95, 22, 127]
# selected_features = [19, 50, 10, 94, 77, 96, 65, 7, 91, 67, 89, 98, 20, 58, 14, 106, 36, 34, 101, 11, 125, 68, 124, 72, 47, 99, 13, 69, 88, 8, 92, 84, 83, 70, 95, 17, 26, 15, 71, 43, 105, 52, 42, 117, 3, 40, 115, 112, 56, 46, 54, 9, 126, 102, 78, 4, 21, 51, 120, 61, 123, 35, 64, 63, 6, 5, 62, 16, 22, 59, 122, 32, 104, 41, 109, 108, 118, 127, 12, 73, 39, 87, 1, 37, 31, 121, 18, 93, 2, 103, 97, 44, 33, 29, 80, 100, 25, 53, 86, 24, 111, 79, 82, 57, 81, 85, 49, 60, 113, 23, 107, 66, 27, 90, 110, 55, 128, 28, 114, 116, 76, 48, 38, 119, 30, 75, 74, 45]
selected_features = [50, 19, 94, 10, 77, 70, 47, 84, 9, 8, 81, 29, 79, 41, 4, 63, 62, 7, 12, 14, 99, 51, 45, 91, 37, 64, 5, 33, 123, 121, 116, 112, 44, 127, 6, 105, 111, 126, 72, 114, 60, 122, 53, 54, 16, 24, 125, 26, 95, 49, 115, 119, 35, 93, 109, 11, 40, 102, 2, 39, 78, 68, 120, 103, 46, 100, 75, 17, 20, 42, 23, 25, 71, 104, 18, 97, 55, 56, 107, 3, 80, 38, 128, 30, 69, 67, 22, 108, 74, 65, 117, 110, 66, 43, 13, 61, 101, 32, 87, 98, 76, 52, 1, 92, 21, 86, 28, 96, 113, 34, 36, 57, 58, 59, 15, 31, 89, 83, 118, 124, 90, 73, 82, 85, 88, 106, 27, 48]
selected_features = [53, 7, 25, 1, 87, 61, 45]

nlme_params = (
		; tvImax=2.1,
		tvIC50=0.4,
		tvKa=1.0,
		Ω=Diagonal([0.2, 0.2, 0.2]),
		σ=0.01
	)

synth_data_pairs = map(1:pop_size) do i
    lv = randn(Float32, LATENT_DIM)
    η = (; η=lv[selected_features[1:3]])

    img = vae.decoder(lv)

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

pop = getindex.(synth_data_pairs, 1)
imgs = getindex.(synth_data_pairs, 2)
lvs = getindex.(synth_data_pairs, 3)
ηs = getindex.(synth_data_pairs, 4)

idx = 13
generated_image = vae.decoder(lvs[idx])
Images.colorview(Gray, generated_image[:, :, 1, 1])


####################################################
# Demonstrate that we can go the back way

# sim = simobs(nlme_model, pop, nlme_params)
# newpop = Subject.(sim)
# plotgrid(newpop[1:12])

fpm = fit(
  nlme_model,
    pop,
    init_params(nlme_model),
    FOCE()
)

fpm
nlme_params

pred = predict(fpm; obstimes=0:0.01:8)

plotgrid(pred[1:8])

η_pred = empirical_bayes(fpm)
ηs
save("saved_data/η_pred.jld2", "η_pred", η_pred)

sim_orig = simobs(nlme_model, pop, nlme_params, ηs; obstimes=0:0.01:10, simulate_error=false)
sim_approx = simobs(nlme_model, pop, nlme_params, η_pred; obstimes=0:0.01:10, simulate_error=false)

plotgrid(sim_orig[1:12]; sim = (; markersize=0))
plotgrid(sim_approx[1:12]; sim = (; markersize=0))




# pop_approx = map(1:pop_size) do i
# 	no_obs_subj = Subject(;
# 		covariates=(; true_η=ηs_approx[i].η, lv=lvs_approx[i]), # Store some relevant info
# 		id=i,
# 		events=DosageRegimen(1.0)
# 	)
# 	sim_approx = simobs(
# 		nlme_model,
# 		no_obs_subj,
# 		nlme_params,
# 		ηs_approx[i];
# 		obstimes=0:0.5:10 # whatever seems appropriate
# 	)
# 	return Subject(sim_approx)
# end
# plotgrid(pop[1:12]; sim=(; markersize=0))
# plotgrid(pop_approx[1:12]; sim=(; markersize=0))
