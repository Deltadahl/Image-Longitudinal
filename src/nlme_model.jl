using Pumas
using DeepPumas
using CairoMakie
using JLD2
include("constants.jl")

pop_size = 100
η_size = 3

datamodel = @model begin
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


p = (
  ; tvImax = 2.1,
  tvIC50 = 0.4,
  tvKa = 1.,
  Ω = Diagonal([0.2, 0.2, 0.2]),
  σ = 0.01
)

dr = DosageRegimen(1., addl=2, ii=2)
_pop = [Subject(; id = i, events = dr, time=0:0.5:8) for i in 1:pop_size]

sim = simobs(datamodel, _pop, p; obstimes=0:0.01:10, simulate_error=false)

η = zero_randeffs(datamodel, _pop, p)

@load "/home/jrun/data/code/TemporalRetinaVAE/output_matrix/testing_output.jld2" latent_vae
@show size(latent_vae)

# vae_η = randn(LATENT_DIM - η_size, length(_pop))
# ηM = randn(η_size, length(_pop))
ηM = zeros(η_size, pop_size)

selected_features = [7, 92, 78]
for (i, feature_nr) in enumerate(selected_features)
    ηM[i, :] = latent_vae[feature_nr, :]
end

η = map(eachindex(_pop)) do i
  (; η = ηM[:, i])
end

sim = simobs(datamodel, _pop, p, η; obstimes=0:0.01:10, simulate_error=false)

plotgrid(sim[1:12]; sim = (; markersize=0))


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
