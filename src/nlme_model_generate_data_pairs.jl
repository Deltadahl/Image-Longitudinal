using JLD2
using Images
include("nlme_model.jl")

η_size = 3

selected_features = [92, 111, 50, 3, 91, 67, 37, 8, 90, 120, 54, 56, 21, 61, 75, 29, 80, 12, 95, 118, 73, 94, 101, 20, 48, 99, 104, 13, 59, 52, 106, 79, 4, 86, 93, 85, 72, 32, 87, 35, 47, 113, 40, 53, 36, 55, 122, 22, 5, 2, 88, 77, 26, 15, 7, 108, 58, 28, 39, 128, 126, 25, 103, 65, 105, 34, 18, 69, 27, 43, 64, 123, 38, 78, 17, 121, 42, 49, 33, 66, 57, 6, 24, 112, 10, 115, 68, 45, 11, 51, 41, 97, 70, 102, 114, 89, 71, 44, 110, 109, 62, 31, 124, 16, 1, 74, 9, 119, 14, 83, 117, 76, 60, 46, 23, 84, 98, 82, 100, 107, 81, 125, 127, 30, 19, 96, 63, 116]

filepath = "../saved_data/eta_approx_and_lv_data_100k.jld2"
dict = load(filepath)
η_approx_matrix = dict["η_approx_matrix"]
lvs_matrix = dict["lvs_matrix"]

pop_size = 100
@time synth_data_pairs = map(1:pop_size) do i
    lv = lvs_matrix[:, i]
    η = (; η=lv[selected_features[1:3]])

    # img = vae.decoder(lv)
    # img = "PLACE HOLDER"
    img = load("../saved_data/imgs_100k/img_$i.png")
    img = Float32.(Gray.(img))
    img = reshape(img, size(img)..., 1, 1)

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
        obstimes=0:0.5:10
    )

    ## Make a Subject of our simulation. The data from no_obs_subj will tag along.
    subj = Subject(sim)
    return (subj, img, lv, η)
end

@time save("../saved_data/synth_data_pairs_XXX.jld2", "synth_data_pairs", synth_data_pairs)
