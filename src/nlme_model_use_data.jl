using JLD2
using Images
# using Plots
using Random
using Statistics
using Bootstrap
using CSV
using DataFrames
include("nlme_model.jl")

# Dictionary to hold all the data
nll_results = Dict()
nll_cis = Dict()

# var_noise = 0.0
for var_noise in [0.0, 1.0, 9.0, 18.0, 49.0]
# for var_noise in [0.0, 1.0]
    pop_size = 100000
    η_size = 3
    obstimes=0:0.5:10
    obstimes_orig=0:0.5:10

    theoretical_max = 1^2 / (1^2 + var_noise)

    base_path = "/mnt/data/code/"
    # filepath = "../data/synthetic/noise_$(var_noise)_eta_pred.jld2"
    filepath = base_path * "predictions/noise_$(var_noise)_eta_pred.jld2"
    η_pred = load(filepath, "η_pred")
    filepath = base_path * "new_data/noise_$(var_noise)_eta_approx_and_lv_data_100k.jld2"

    η_approx = load(filepath, "η_approx")
    η_true = load(filepath, "η_true_noise")
    η_true_no_noise = load(filepath, "η_true")

    println("mean η_true $(mean(η_true))")
    println("mean η_true_no_noise $(mean(η_true_no_noise))")
    println("mean η_approx $(mean(η_approx))")
    println("mean η_pred $(mean(η_pred))")
    println()
    println("variance η_true $(var(η_true))")
    println("variance η_true_no_noise $(var(η_true_no_noise))")
    println("variance η_approx $(var(η_approx))")
    println("variance η_pred $(var(η_pred))")


    function get_pop(eta_fun)
        pop = map(1:pop_size) do i
            η = (; η=eta_fun(i))

            ## Create a subject that just stores some covariates and a dosing regimen
            no_obs_subj = Subject(;
                covariates=(;true_η=η.η), # Store some info
                id=i,
                events=DosageRegimen(1.0)
            )

            ## Simulate some observations for the subject
            sim = simobs(
                nlme_model,
                no_obs_subj,
                nlme_params,
                η;
                obstimes=obstimes
            )

            ## Make a Subject of our simulation. The data from no_obs_subj will tag along.
            subj = Subject(sim)
            return subj
            # return (subj, img, lv, η)
        end
    end

    function eta_fun_true(i)
        return η_true[:, i]
    end

    function eta_fun_true_no_noise(i)
        return η_true_no_noise[:, i]
    end

    function eta_fun_pred(i)
        return η_pred[:, i]
    end

    function eta_fun_approx(i)
        return η_approx[:, i]
    end

    function eta_fun_average(i)
        return zeros(3)
    end

    function eta_fun_random(i)
        return randn(3)
    end

    pop = get_pop(eta_fun_true)
    pop_no_noise = get_pop(eta_fun_true_no_noise)
    pop_pred = get_pop(eta_fun_pred)
    pop_approx = get_pop(eta_fun_approx)
    pop_average = get_pop(eta_fun_average)
    pop_random = get_pop(eta_fun_random)

    i = 1
    sub = pop[i]
    sub_pred = pop_pred[i]
    sub_average = pop_average[i]
    id = sub.id
    dose_time = sub.events[1].time
    data = sub.observations.Outcome
    data_pred = sub_pred.observations.Outcome
    data_average = sub_average.observations.Outcome

    ####

    # p1 = Plots.plot(obstimes_orig, data, title="ID: $id", markershape=:circle, label="True data", color=:black)
    # p1 = plot(obstimes_orig, data, title="ID: $id", markershape=:circle, label="True data", color=:black)
    # plot!(obstimes, data_pred, markershape=:circle, label="Predicted data", color=:red)
    # plot!(obstimes, data_average, markershape=:circle, label="Average data", color=:orange)
    # xlabel!(p1, "Time")
    # ylabel!(p1, "Outcome")
    # vline!(p1, [dose_time], linestyle=:dot, color=:gray, label="Dose", lw=2)
    # # Plots.plot!(obstimes, data_pred, markershape=:circle, label="Predicted data", color=:red)
    # # Plots.plot!(obstimes, data_average, markershape=:circle, label="Average data", color=:orange)
    # # Plots.xlabel!(p1, "Time")
    # # Plots.ylabel!(p1, "Outcome")
    # # Plots.vline!(p1, [dose_time], linestyle=:dot, color=:gray, label="Dose", lw=2)

    function get_mse_long(pop1, pop2)
        mse_counter = 0
        for i = 1:pop_size
            mse_counter += mean((pop1[i].observations.Outcome .- pop2[i].observations.Outcome).^2)
        end
        mse_counter /= pop_size
    end

    get_mse_long(pop, pop_pred)
    get_mse_long(pop, pop_approx)
    get_mse_long(pop, pop_average)
    get_mse_long(pop, pop_random)
    get_mse_long(pop, pop_no_noise)


    function export_pop_data(populations::Dict{String,<:Any}, var_name::String)
        for (pop_name, population) in populations
            filename = filename = "/home/jrun/data/code/longitudinal_data/noise_$(var_name)_$(pop_name).csv"
            df = DataFrame()  # Initialize empty DataFrame
            for i in 1:length(population)
                # Extract the outcome data for the i-th individual in the current population
                outcomes = population[i].observations.Outcome
                # Append to DataFrame: add a new column for each individual
                df[!, "Individual_$i"] = outcomes
            end
            # Write DataFrame to CSV

            CSV.write(filename, df)
            println("Data written to $filename")
        end
    end

    populations = Dict(
        "pop" => pop,
        "pop_no_noise" => pop_no_noise,
        "pop_pred" => pop_pred,
        "pop_approx" => pop_approx,
        "pop_average" => pop_average,
        "pop_random" => pop_random
    )

    export_pop_data(populations, string(var_noise))


    # # Calculate the R^2 correlation for the longitudinal data
    # function get_r2_long(pop1, pop2)
    #     SS_res = 0.0
    #     SS_tot = 0.0
    #     pop_size = length(pop1)  # Assuming pop1 and pop2 have the same length

    #     # Calculate the mean of the true data
    #     y_bar = mean([mean(subpop.observations.Outcome) for subpop in pop1])

    #     for i = 1:pop_size
    #         # Residual sum of squares
    #         SS_res += sum((pop1[i].observations.Outcome .- pop2[i].observations.Outcome).^2)

    #         # Total sum of squares
    #         SS_tot += sum((pop1[i].observations.Outcome .- y_bar).^2)
    #     end

    #     # Compute R^2
    #     r2 = 1 - (SS_res / SS_tot)

    #     return (r2, r2/theoretical_max)
    # end

    # get_r2_long(pop, pop_pred)
    # get_r2_long(pop, pop_approx)
    # get_r2_long(pop, pop_average)
    # get_r2_long(pop, pop_random)

    # Calculate the R^2 correlation for the random effects
    function get_r2_η(η_real, η_compare)
        SS_res = 0.0
        SS_tot = 0.0

        # Calculate the mean of the true data
        η_bar = mean([mean(η_sub) for η_sub in η_real])

        for i = 1:pop_size
            # Residual sum of squares
            SS_res += sum((η_real[i] .- η_compare[i]).^2)

            # Total sum of squares
            SS_tot += sum((η_real[i] .- η_bar).^2)
        end

        # Compute R^2
        r2 = 1 - (SS_res / SS_tot)

        return (r2, r2/theoretical_max)
    end

    η_zero = zeros(size(η_pred))
    η_random = randn(size(η_pred))

    get_r2_η(η_true, η_pred)
    get_r2_η(η_true, η_true_no_noise)
    get_r2_η(η_true, η_approx)
    get_r2_η(η_true, η_zero)
    get_r2_η(η_true, η_random)


    function compute_conditional_nll(population, params)
        nlls = Float64[]
        for (i, subj) in enumerate(population)
            # Ensure randeffs are correctly mapped to NamedTuple expected by Pumas
            # randeffs_nt = (η = η_true[:, i],)
            randeffs_nt = (η = η_true_no_noise[:, i],)
            nll = conditional_nll(nlme_model, subj, params, randeffs_nt)
            push!(nlls, nll)
        end
        return nlls
    end

    nll_true = compute_conditional_nll(pop, nlme_params)
    nll_pred = compute_conditional_nll(pop_pred, nlme_params)
    nll_approx = compute_conditional_nll(pop_approx, nlme_params)
    nll_average = compute_conditional_nll(pop_average, nlme_params)
    nll_random = compute_conditional_nll(pop_random, nlme_params)

    println("NLL for true random effects: ", mean(nll_true))
    println("NLL for predicted random effects: ", mean(nll_pred))
    println("NLL for approximate random effects: ", mean(nll_approx))
    println("NLL for average random effects: ", mean(nll_average))
    println("NLL for random random effects: ", mean(nll_random))

    function get_bootstrap_nll(nll_values::Vector)
        bootstrap_results = bootstrap(mean, nll_values, BasicSampling(1000))  # 1000 is the number of bootstrap samples

        # Extract the bootstrap confidence interval
        bootstrap_ci = confint(bootstrap_results, BasicConfInt(0.95))

        return bootstrap_ci
    end
    nll_true_ci = get_bootstrap_nll(nll_true)
    nll_pred_ci = get_bootstrap_nll(nll_pred)
    nll_approx_ci = get_bootstrap_nll(nll_approx)
    nll_average_ci = get_bootstrap_nll(nll_average)
    nll_random_ci = get_bootstrap_nll(nll_random)


    current_noise_level = var_noise
    nll_results[current_noise_level] = Dict(
        "True" => mean(nll_true),
        "Predicted" => mean(nll_pred),
        "Approximate" => mean(nll_approx),
        "Average" => mean(nll_average),
        "Random" => mean(nll_random)
    )

    nll_cis[current_noise_level] = Dict(
        "True" => nll_true_ci,
        "Predicted" => nll_pred_ci,
        "Approximate" => nll_approx_ci,
        "Average" => nll_average_ci,
        "Random" => nll_random_ci
    )

end



function generate_latex_table(nll_results, nll_cis)
    sorted_keys = sort(collect(keys(nll_results)))
    latex_str = "\\begin{table}[h]\n\\centering\n\\caption{Conditional NLL for Different Noise Levels (95\\% CI)}\n\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline\n"
    latex_str *= "Noise Level & True & Predicted & Approximate & Average & Random \\\\ \\hline\n"

    for noise_level in sorted_keys
        latex_str *= "$(noise_level) "
        for effect_type in ["True", "Predicted", "Approximate", "Average", "Random"]
            mean_val = round(Int, nll_results[noise_level][effect_type])
            ci = nll_cis[noise_level][effect_type][1]
            ci_lower = round(Int, ci[2])
            ci_upper = round(Int, ci[3])
            latex_str *= "& $(mean_val) [$(ci_lower), $(ci_upper)] "
        end
        latex_str *= "\\\\ \\hline\n"
    end

    latex_str *= "\\end{tabular}\n\\label{tab:nll_results}\n\\end{table}"
    return latex_str
end

println(generate_latex_table(nll_results, nll_cis))



# ############################################
# # Set the number of bootstrap samples
# n_bootstraps = 1000

# # Function to calculate MSE on bootstrap sample
# function mse(x::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
#     diff_squared = [((x[i][1][j] - x[i][2][j])^2) for i in 1:length(x) for j in 1:length(x[i][1])]
#     return mean(diff_squared)
# end

# function get_bootstrap(pop1::Vector, pop2::Vector)
#     # Combine populations into one vector of tuples, where each tuple contains corresponding vectors of outcomes from pop1 and pop2
#     pop_vector = [(pop1[i].observations.Outcome, pop2[i].observations.Outcome) for i in 1:length(pop1)]

#     # Perform the bootstrap
#     bootstrap_results = bootstrap(mse, pop_vector, BasicSampling(n_bootstraps))

#     # Extract the bootstrap confidence interval
#     bootstrap_ci = confint(bootstrap_results, BasicConfInt(0.95))

#     return bootstrap_ci
# end

# @show get_bootstrap(pop[1:pop_size], pop_pred)
# @show get_bootstrap(pop[1:pop_size], pop_approx)
# @show get_bootstrap(pop[1:pop_size], pop_average)
# @show get_bootstrap(pop[1:pop_size], pop_random)



# mean((η_true .- η_pred).^2)
# mean((η_true .- η_zero).^2)
# mean((η_true .- η_random).^2)

# mse(x::Matrix) = mean((x[:,1] .- x[:,2]).^2)
# n_bootstraps = 1000
# Random.seed!(1)
# function get_bootstrap(η_1, η_2)
#     η_matrix = hcat(vec(η_1), vec(η_2))
#     bootstrap_results = bootstrap(mse, η_matrix, BasicSampling(n_bootstraps))
#     bootstrap_ci = confint(bootstrap_results, BasicConfInt(0.95))
# end
# # @show get_bootstrap(η_true, η_pred)
# # @show get_bootstrap(η_true, η_zero)
# # @show get_bootstrap(η_true, η_random)

# function plot_data()
#     Width = 3
#     Height = 4

#     p = Plots.plot(layout=(Height, Width), size=(800, 800))

#     # Determine the global y-axis range
#     global_ymax = maximum([maximum(sub.observations.Outcome) for sub in pop[1:Width*Height]])

#     data_average = sub_average.observations.Outcome

#     for j in 1:Height
#         for k in 1:Width
#             i = (j-1)*Width + k  # Calculate i based on j and k

#             sub = pop[i]
#             sub_pred = pop_pred[i]
#             id = sub.id
#             dose_time = sub.events[1].time
#             data = sub.observations.Outcome
#             data_pred = sub_pred.observations.Outcome

#             color1 = :blue
#             color2 = :red
#             color3 = :orange
#             linewidth=1.4

#             Plots.plot!(p[j,k], obstimes_orig, data, title="ID: $id", titlefontsize=8, label=(k == ceil(Int, Width) && j == 1 ? "True data" : ""), legend=false, ylim=(-0.05, global_ymax+0.05), grid=:both, linewidth=linewidth, linecolor=color1)
#             Plots.plot!(p[j,k], obstimes, data_pred, label=(k == ceil(Int, Width) && j == 1 ? "Predicted data" : ""), linecolor=color2, linewidth=linewidth)
#             Plots.plot!(p[j,k], obstimes, data_average, label=(k == ceil(Int, Width) && j == 1 ? "Average data" : ""), linecolor=color3, linewidth=linewidth)

#             # Label the bottom left x-axis as "Time"
#             if j == Height && k == 1
#                 Plots.xlabel!(p[j,k], "Time")
#             elseif j != Height
#                 Plots.plot!(p[j,k], xticks=:grid)
#             end

#             # Label the bottom left y-axis as "Outcome"
#             if k == 1 && j == Height
#                 Plots.ylabel!(p[j,k], "Outcome")
#             elseif k != 1
#                 Plots.plot!(p[j,k], yticks=:grid)
#             end

#             Plots.vline!(p[j,k], [dose_time], linestyle=:dash, color=:gray, label=(k == ceil(Int, Width) && j == 1 ? "Dose" : ""), lw=1)
#         end
#     end

#     # The "invisible" plot to make legend
#     Plots.plot!(p[1, ceil(Int, Width)], [], [], label="", legend=:topright, alpha=0.0, legendfontsize=10)
#     savefig("../saved_figures/comparison_12_3.png")
#     Plots.display(p)
# end

# plot_data()
