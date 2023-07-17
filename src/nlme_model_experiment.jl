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
        tvImax ∈ RealDomain(; lower=0.0, init=1.1)
        tvIC50 ∈ RealDomain(; lower=0.0, init=0.6)
        tvKa ∈ RealDomain(; lower=0.0)
        Ω ∈ PDiagDomain(; init=Diagonal([0.2, 0.2, 0.2]))
        σ ∈ RealDomain(; lower=0.0, init=0.1)
    end
    @random η ~ MvNormal(Ω)
    @pre begin
        Ka = tvKa * exp(η[1])
        Imax = tvImax * exp(η[2])
        IC50 = tvIC50 * exp(η[3])
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

save_nr = 1895
model_path = "../saved_models/save_nr_$(save_nr).jld2"
vae = load(model_path, "vae")
# vae_copy = load(model_path, "vae")
# vae = vae_copy

selected_features = [75, 25, 98, 18, 107, 101, 33, 60, 52, 34, 47, 116, 102, 5, 113, 43, 128, 14, 41, 16, 38, 12, 4, 115, 105, 26, 64, 32, 1, 103, 59, 63, 21, 56, 31, 72, 46, 36, 108, 104, 89, 82, 124, 97, 122, 53, 84, 55, 57, 65, 6, 94, 39, 15, 78, 74, 49, 50, 96, 7, 83, 81, 93, 17, 20, 69, 87, 125, 112, 28, 90, 110, 40, 117, 2, 99, 10, 88, 118, 29, 30, 42, 23, 119, 67, 19, 44, 126, 70, 73, 45, 114, 35, 106, 86, 109, 85, 77, 71, 62, 54, 51, 13, 91, 123, 66, 27, 58, 61, 68, 11, 37, 76, 48, 120, 8, 3, 80, 79, 121, 100, 92, 24, 9, 111, 95, 22, 127]

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

lvs_approx = similar(lvs)
for (i, img) in enumerate(imgs)
    lvs_approx[i] = vae.μ_layer(vae.encoder(img))[:, 1]
end

ηs_approx = map(eachindex(lvs_approx)) do i
    (; η=lvs_approx[i][selected_features[1:3]])
end

idx = 5
ηs_approx[idx].η
ηs[idx].η


# idx = 7
# generated_image = vae.decoder(lvs[idx])
# Images.colorview(Gray, generated_image[:, :, 1, 1])
# generated_image = vae.decoder(lvs_approx[idx])
# Images.colorview(Gray, generated_image[:, :, 1, 1])

# plotgrid(pop[1:12]; sim=(; markersize=0))

idx = 1
pop_approx = map(1:pop_size) do i
	no_obs_subj = Subject(;
		covariates=(; true_η=ηs_approx[i].η, lv=lvs_approx[i]), # Store some relevant info
		id=i,
		events=DosageRegimen(1.0)
	)
	sim_approx = simobs(
		nlme_model,
		no_obs_subj,
		nlme_params,
		ηs_approx[i];
		obstimes=0:0.5:10 # whatever seems appropriate
	)
	return Subject(sim_approx)
end
plotgrid(pop[1:12]; sim=(; markersize=0))
plotgrid(pop_approx[1:12]; sim=(; markersize=0))

####################################################
# Demonstrate that we can go the back way

# sim = simobs(nlme_model, pop, nlme_params)
# newpop = Subject.(sim)

# plotgrid(newpop[1:12])

model = @model begin
    @param begin
        tvImax ∈ RealDomain(; lower=0.0)
        tvIC50 ∈ RealDomain(; lower=0.0)
        tvKa ∈ RealDomain(; lower=0.0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0.0)
    end
    @random η ~ MvNormal(Ω)
    @pre begin
        Ka = tvKa * exp(η[1])
        Imax = tvImax * exp(η[2])
        IC50 = tvIC50 * exp(η[3])
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end


fpm = fit(
    nlme_model,
    newpop,
    init_params(nlme_model),
    FOCE()
)

fpm
p

pred = predict(fpm; obstimes=0:0.01:8)

plotgrid(pred[1:8])

η_pred = empirical_bayes(fpm)
