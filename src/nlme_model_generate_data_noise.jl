using Pumas
using DeepPumas
using CairoMakie
using JLD2
using Images
include("constants.jl")
include("nlme_model.jl")

pop_size = 1_000_000
η_size = 3
selected_features = [92, 111, 50, 3, 91, 67, 37, 8, 90, 120, 54, 56, 21, 61, 75, 29, 80, 12, 95, 118, 73, 94, 101, 20, 48, 99, 104, 13, 59, 52, 106, 79, 4, 86, 93, 85, 72, 32, 87, 35, 47, 113, 40, 53, 36, 55, 122, 22, 5, 2, 88, 77, 26, 15, 7, 108, 58, 28, 39, 128, 126, 25, 103, 65, 105, 34, 18, 69, 27, 43, 64, 123, 38, 78, 17, 121, 42, 49, 33, 66, 57, 6, 24, 112, 10, 115, 68, 45, 11, 51, 41, 97, 70, 102, 114, 89, 71, 44, 110, 109, 62, 31, 124, 16, 1, 74, 9, 119, 14, 83, 117, 76, 60, 46, 23, 84, 98, 82, 100, 107, 81, 125, 127, 30, 19, 96, 63, 116]
pwd()
# Using for test data
dict_synth = load("noise_0.0_eta_approx_and_lv_data_1000k.jld2")
println(dict_synth.keys)
lvs_mat = getindex.(dict_synth["lvs_matrix"])
η_true = getindex.(dict_synth["η_true"])

#test

# lvs_mat_loaded = getindex.(dict_synth["synth_data_pairs"], 3);
# ηs_loaded = getindex.(dict_synth["synth_data_pairs"], 4);
# η_true = hcat([t.η for t in ηs_loaded]...)
# lvs_mat = hcat([t for t in lvs_mat_loaded]...)

# Using for train and validation data data
# lvs_mat = randn(Float32, (LATENT_DIM, pop_size))
# η_true = lvs_mat[selected_features[1:3], :]

var_true = var(η_true)
var_noise = 0.0
r = randn(size(η_true))
r .*= sqrt(var_noise)
var_r = var(r)
explainable = var_true / (var_true + var_r)
η_matrix_noise = η_true .+ r
var(η_matrix_noise)
η_matrix_noise ./= std(η_matrix_noise)
var(η_matrix_noise)


@time synth_data_pairs = map(1:pop_size) do i
  lv = lvs_mat[:, i]
  η = (; η=η_matrix_noise[:, i])

  # img = vae.decoder(lv)
  img = "PLACE HOLDER"
  # img = load("../saved_data/imgs_100k/img_$i.png")
  # img = Float32.(Gray.(img))
  # img = reshape(img, size(img)..., 1, 1)

  ## Create a subject that just stores some covariates and a dosing regimen
  no_obs_subj = Subject(;
      covariates=(; img, true_η=η.η, lv=lv),
      id=i,
      events=DosageRegimen(1.0)
  )

  ## Simulate some observations for the subject
  sim = simobs(
      nlme_model,
      no_obs_subj,
      nlme_params,
      η;
      obstimes=0:0.5:10
  )

  ## Make a Subject of our simulation. The data from no_obs_subj will tag along.
  subj = Subject(sim)
  return (subj, img, lv, η)
end

pop = getindex.(synth_data_pairs, 1);
imgs = getindex.(synth_data_pairs, 2);
lvs = getindex.(synth_data_pairs, 3);
ηs = getindex.(synth_data_pairs, 4);

####################### Back way #############################

@time fpm = fit(
  nlme_model,
    pop,
    init_params(nlme_model),
    FOCE()
)

@time η_approx = empirical_bayes(fpm)

η_approx_matrix = Float32.(hcat([t.η for t in η_approx]...))
η_true_noise = hcat([t.η for t in ηs]...)

error_fit = mean((η_approx_matrix .- η_true_noise).^2)
error_true = mean((η_approx_matrix .- η_true).^2)
error_zero = mean((zeros(size(η_true_noise)) .- η_true_noise).^2)
error_random = mean((randn(size(η_true_noise)) .- η_true_noise).^2)

lvs_matrix = hcat(lvs...)
save("new_data/noise_$(var_noise)_eta_approx_and_lv_data_$(Int(pop_size/1000))k.jld2", "η_approx", η_approx_matrix, "lvs_matrix", lvs_matrix, "η_true", η_true, "η_true_noise", η_true_noise)
