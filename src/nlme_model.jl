using Pumas
using DeepPumas
using CairoMakie
using JLD2
include("constants.jl")


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
_pop = [Subject(; id = i, events = dr, time=0:0.5:8) for i in 1:100]


sim = simobs(datamodel, _pop, p; obstimes=0:0.01:10, simulate_error=false)


η = zero_randeffs(datamodel, _pop, p)

num_rand_effects = 3
# load "output_matrix/testing_output.jld2" result
@load "output_matrix/testing_output.jld2" result

vae_η = randn(LATENT_DIM - num_rand_effects, length(_pop))
ηM = randn(num_rand_effects, length(_pop))
# fill ηM with 2.2958581   0.1355047   0.17932206
for i = 1:length(_pop)
  ηM[:, i] = [2.2958581, 0.1355047, 0.17932206]
end

vae_features = vcat(vae_η, ηM)

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
