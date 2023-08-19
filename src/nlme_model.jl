using Pumas
using DeepPumas
using CairoMakie

min_val = 1.0 * 10^(-8)
nlme_model = @model begin
  @param begin
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
    Central' = Ka * Depot - Imax * Central / (IC50 + Central)
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end

nlme_params = (
		; tvImax=2.1,
		tvIC50=0.4,
		tvKa=1.0,
		Ω=Diagonal([1.0, 1.0, 1.0]),
		σ=0.01
	)
