using Pumas
using DeepPumas
using CairoMakie
using JLD2
using Images
include("constants.jl")

pop_size = 1_00
η_size = 3
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

# save_nr = 526
# model_path = "../saved_models/save_nr_$(save_nr).jld2"
# vae = load(model_path, "vae") # Use when creating images.

selected_features = [92, 111, 50, 3, 91, 67, 37, 8, 90, 120, 54, 56, 21, 61, 75, 29, 80, 12, 95, 118, 73, 94, 101, 20, 48, 99, 104, 13, 59, 52, 106, 79, 4, 86, 93, 85, 72, 32, 87, 35, 47, 113, 40, 53, 36, 55, 122, 22, 5, 2, 88, 77, 26, 15, 7, 108, 58, 28, 39, 128, 126, 25, 103, 65, 105, 34, 18, 69, 27, 43, 64, 123, 38, 78, 17, 121, 42, 49, 33, 66, 57, 6, 24, 112, 10, 115, 68, 45, 11, 51, 41, 97, 70, 102, 114, 89, 71, 44, 110, 109, 62, 31, 124, 16, 1, 74, 9, 119, 14, 83, 117, 76, 60, 46, 23, 84, 98, 82, 100, 107, 81, 125, 127, 30, 19, 96, 63, 116]

nlme_params = (
		; tvImax=2.1,
		tvIC50=0.4,
		tvKa=1.0,
		Ω=Diagonal([1.0, 1.0, 1.0]),
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
ηs

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
# end1

@time η_approx = empirical_bayes(fpm)
ηs

η_approx_matrix = Float32.(hcat([t.η for t in η_approx]...))
# η_approx_matrix2 = hcat([t.η for t in ebes]...)
η_true = hcat([t.η for t in ηs]...)

error_fit = mean((η_approx_matrix .- η_true).^2)
# error_ebes = mean((η_approx_matrix2 .- η_true).^2)
error_random = mean((randn(size(η_true)) .- η_true).^2)
error_zero = mean((zeros(size(η_true)) .- η_true).^2)

lvs_matrix = hcat(lvs...)
save("eta_approx_and_lv_data_XXX.jld2", "η_approx_matrix", η_approx_matrix, "lvs_matrix", lvs_matrix)
# save("eta_and_img_data.jld2", "η_approx_matrix", η_approx_matrix, "imgs", imgs)
# save("eta_and_lv_data.jld2", "η_approx_matrix", η_approx_matrix, "lvs", lvs)

# stderror(infer(fpm))
