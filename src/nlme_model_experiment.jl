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
    tvImax ∈ RealDomain(; lower=0., init=1.1)
    tvIC50 ∈ RealDomain(; lower=0., init=0.6)
    tvKa ∈ RealDomain(; lower=0.)
    Ω ∈ PDiagDomain(; init = Diagonal([0.2, 0.2, 0.2]))
    σ ∈ RealDomain(; lower=0., init=0.1)
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


# p = (
#   ; tvImax = 2.1,
#   tvIC50 = 0.4,
#   tvKa = 1.,
#   Ω = Diagonal([0.2, 0.2, 0.2]),
#   σ = 0.01
# )

# dr = DosageRegimen(1., addl=2, ii=2)
# _pop = [Subject(; id = i, events = dr, time=0:0.5:8) for i in 1:pop_size]

# sim = simobs(nlme_model, _pop, p; obstimes=0:0.01:10, simulate_error=false)

pwd()
cd("/home/jrun/data/code/TemporalRetinaVAE/src")
pwd()
save_nr = 518
model_path = "../saved_models/save_nr_$(save_nr).jld2"
vae_copy = load(model_path, "vae")
vae = vae_copy

# vae.encoder = vae.encoder |> DEVICE
# vae.μ_layer = vae.μ_layer |> DEVICE
# vae.logvar_layer = vae.logvar_layer |> DEVICE
# vae.decoder = vae.decoder |> DEVICE
# vae = vae |> DEVICE
selected_features = [41, 118, 105] # TODO change to correct values (find the most impactful)

synth_data_pairs = map(1:pop_size) do i
  lv = randn(Float32, LATENT_DIM)
  η = (; η = lv[selected_features])

  img = vae.decoder(lv)

  ## Create a subject that just stores some covariates and a dosing regimen
  no_obs_subj = Subject(;
    covariates=(; img, true_η=η.η, lv=lv), # Store some relevant info
    id=i,
    events=DosageRegimen(1.),
  )
  nlme_params = (
    ; tvImax = 2.1,
    tvIC50 = 0.4,
    tvKa = 1.,
    Ω = Diagonal([0.2, 0.2, 0.2]),
    σ = 0.01
    )

  ## Simulate some observations for the subject
  sim = simobs(
      nlme_model,
      no_obs_subj,
      nlme_params,
      η;
      obstimes = 0:0.5:10 # whatever seems appropriate
    )

  ## Make a Subject of our simulation. The data from no_obs_subj will tag along.
  subj = Subject(sim)
  return (subj, img, lv, η)
end

synth_pop = getindex.(synth_data_pairs, 1)
synth_imgs = getindex.(synth_data_pairs, 2)
synth_lvs = getindex.(synth_data_pairs, 3)
synth_ηs = getindex.(synth_data_pairs, 4)

ind_idx = 10
typeof(synth_pop[ind_idx])
typeof(synth_imgs[ind_idx])
size(synth_imgs[ind_idx])
generated_image = cpu(synth_imgs[ind_idx])
generated_image = generated_image[:,:,1,1]
generated_image = Images.colorview(Gray, generated_image)
path_to_save = joinpath("/home/jrun/data/code/TemporalRetinaVAE", "test.png")
save(path_to_save, generated_image)

plotgrid(synthpop[1:12]; sim = (; markersize=0))
# plotgrid(sim[1:12]; sim = (; markersize=0))

encoded = vae.encoder(synth_imgs[ind_idx])
lv_estimated = vae.μ_layer(encoded)

η_estimated = (; η = lv_estimated[selected_features])
synth_ηs[1]


# vae.decoder(randn(Float32, (LATENT_DIM)))
# vae.encoder(randn(Float32, (224,224,1,1)))
####################################################
# Demonstrera att vi kan gå bakvägen

sim = simobs(datamodel, _pop, p)
newpop = Subject.(sim)

plotgrid(newpop[1:12])

model = @model begin
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


fpm = fit(
  model,
  newpop,
  init_params(model),
  FOCE()
)

fpm
p

pred = predict(fpm; obstimes=0:0.01:8)

plotgrid(pred[1:8])

η_pred = empirical_bayes(fpm)
