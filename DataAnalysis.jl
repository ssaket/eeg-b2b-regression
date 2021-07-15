### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 04ddbcb7-5c2b-4c3b-8cf5-a5b294af8d64
using Unfold, StatsModels

# ╔═╡ 4493bf65-5fec-4be3-bff2-491c2562ce78
include("src/utils.jl")

# ╔═╡ 369e1538-3855-4d0b-9b17-6f7dd32e9cc9
md"""
# Encoding and Decoding Analysis using Back-to-Back regression 
"""

# ╔═╡ bf87faea-f9b3-4403-867a-422a57034d31
md"""
## Introduction
"""

# ╔═╡ 6eb82b93-d1b8-4c5a-beaf-892f40560fd8
using CairoMakie
CairoMakie.activate!()

# ╔═╡ 7f6c9361-dbf2-4fed-88f6-71c2add543aa
md"""
### load data and events
"""

# ╔═╡ 24e747ea-792c-41d1-9fba-eb7117ccce3a
function load_data(sub = "sub-45")
	path = "data/$(sub)/eeg/$(sub)_task-WLFO_eeg.set"
	!isfile(path) && (path = "/store/data/WLFO/derivatives/preproc_agert/$(sub)/eeg/$(sub)_task-WLFO_eeg.set")
	data, events = read_eeglab(path, 128)	
	return (data, events)
end

# ╔═╡ 469f905a-4e65-4252-a142-3ad8bcd47854
md"""
Filter events types to include only *Fixations*
"""

# ╔═╡ 1a2bdc3f-3faa-48db-9247-98b30b4da23e
begin
	data, events = load_data("sub-45")
	events = events[events.type .== "fixation",:]
end

# ╔═╡ f6cf5ad3-5aad-449b-b563-e7f1a1db7c43
begin
	beta, times = Unfold.epoch(data=data, tbl=events, τ=(-0.3, 0.5), sfreq=128)
	beta[:,:,isnan.(events.sac_vmax)] .= missing # to run with sac_vmax
	beta[:,:,isnan.(events.fix_avgpos_x)] .= missing # to run with sac_vmax
end

# ╔═╡ cd36c14d-fc59-4327-b807-776f007b1f10
function get_gamma(f, events, beta, times)
	se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y,cross_val_reps = 5)
	model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)
	return(model, results_expanded)
end
	

# ╔═╡ c6c154d4-0a74-4fa9-b0ad-5c52b9377c2d
md"""
## Observations
"""

# ╔═╡ 62f9cb36-1732-4b59-9013-eda6b3f8237e
md"""
### Ground truth data
"""

# ╔═╡ 0b361dee-8e1e-4ddc-807d-789de6d5e498
md"""
we plot the gamma of Intercept and Saccades amplitude.\
$y = \beta_0 + \beta_1 * sacc\_amp$
"""

# ╔═╡ 5524c3fe-0c6d-48df-80dc-98ddf5172515
begin
	f_amp = @formula 0~1 + sac_amplitude; 
	m, res_amp = get_gamma(f_amp, events, beta, times);
	pt_amp = plot_results(res_amp,layout_x=:basisname);
end

# ╔═╡ 044c89eb-9b08-41ab-a4f6-06dd63f35df2
md"""

we plot the gamma of Intercept and Saccades Velocity.\
$y = \beta_0 + \beta_1 * sacc\_vmax$

"""

# ╔═╡ 7ad9d3d7-d77b-483b-8e26-bed435345ffd
begin
	f_vmax = @formula 0~1 + sac_vmax
	_, res_vmax = get_gamma(f_vmax, events, beta, times);
	pt_vmax = plot_results(res_vmax,layout_x=:basisname)
end

# ╔═╡ 8b79b4e0-8f0f-49ba-b29b-6f5556897847
md"""
we plot the gamma of Intercept, Saccades amplitude and Saccades Velocity.\
$y = \beta_0 + \beta_1 * sacc\_amp + \beta_2 * sacc\_vmax$
"""

# ╔═╡ 31811918-12b5-4540-b6c7-b18ce88e111d
begin
	f_amp_vmax = @formula 0~1 + sac_amplitude + sac_vmax
	_, res_amp_vmax = get_gamma(f_amp_vmax, events, beta, times);
	pt_amp_vmax = plot_results(res_amp_vmax,layout_x=:basisname)
end
	

# ╔═╡ 54a02c16-48f4-4bdc-ba06-3d6af0a971d7
md"""
we plot the gamma of Intercept, human face, saccades amplitude and saccades velocity.\
$y = \beta_0 + \beta_1 * cat(human\_face) + \beta_2 * sac\_amplitude + \beta_3 * sac\_vmax$
"""

# ╔═╡ c5dc110c-240c-423e-a660-42644f5c7fcb
begin
	f_amp_vmax_hf = @formula 0~1 + sac_amplitude + sac_vmax + humanface
	_, res_amp_vmax_hf = get_gamma(f_amp_vmax_hf, events, beta, times);
	pt_amp_vmax_hf = plot_results(res_amp_vmax_hf,layout_x=:basisname)
end

# ╔═╡ 61695ced-5892-47bf-9cde-af3453658211
begin
	scene = vbox(pt_amp, pt_vmax)
	scene
end

# ╔═╡ ae0d9783-726d-4b07-a616-c4980935b3f9
md"""

# Playground

"""

# ╔═╡ c6020209-bb67-4b6d-b813-ac0c24d973b4
md"""
## Using B2BRegression Module
"""

# ╔═╡ 5d25d703-f49b-4975-baee-d71cf80203fa
function get_path(sub)
	path = "data/$(sub)/eeg/$(sub)_task-WLFO_eeg.set"
	if !isfile(path)
		path = "/store/data/WLFO/derivatives/preproc_agert/$(sub)/eeg/$(sub)_task-WLFO_eeg.set"
	end
	return path
end

# ╔═╡ 9bb9ae0d-5149-492d-af64-e33c7f4707fe
path = get_path("sub-45")

# ╔═╡ 0ada22b0-fa5f-4d77-a806-96039e72fb10
B2BRegression.plot_massunivariate_gamma(
	path, 
	(@formula 0~1 + sac_amplitude), 
	τ = (-0.3, 0.5),
	event_types = ["fixation"],
	sfreq = 128,
    cross_val_reps = 5,
    mixed = false
)

# ╔═╡ Cell order:
# ╟─369e1538-3855-4d0b-9b17-6f7dd32e9cc9
# ╟─bf87faea-f9b3-4403-867a-422a57034d31
# ╠═04ddbcb7-5c2b-4c3b-8cf5-a5b294af8d64
# ╠═6eb82b93-d1b8-4c5a-beaf-892f40560fd8
# ╠═4493bf65-5fec-4be3-bff2-491c2562ce78
# ╟─7f6c9361-dbf2-4fed-88f6-71c2add543aa
# ╠═24e747ea-792c-41d1-9fba-eb7117ccce3a
# ╟─469f905a-4e65-4252-a142-3ad8bcd47854
# ╠═1a2bdc3f-3faa-48db-9247-98b30b4da23e
# ╠═f6cf5ad3-5aad-449b-b563-e7f1a1db7c43
# ╠═cd36c14d-fc59-4327-b807-776f007b1f10
# ╟─c6c154d4-0a74-4fa9-b0ad-5c52b9377c2d
# ╟─62f9cb36-1732-4b59-9013-eda6b3f8237e
# ╟─0b361dee-8e1e-4ddc-807d-789de6d5e498
# ╠═5524c3fe-0c6d-48df-80dc-98ddf5172515
# ╟─044c89eb-9b08-41ab-a4f6-06dd63f35df2
# ╠═7ad9d3d7-d77b-483b-8e26-bed435345ffd
# ╟─8b79b4e0-8f0f-49ba-b29b-6f5556897847
# ╠═31811918-12b5-4540-b6c7-b18ce88e111d
# ╟─54a02c16-48f4-4bdc-ba06-3d6af0a971d7
# ╠═c5dc110c-240c-423e-a660-42644f5c7fcb
# ╠═61695ced-5892-47bf-9cde-af3453658211
# ╟─ae0d9783-726d-4b07-a616-c4980935b3f9
# ╠═c6020209-bb67-4b6d-b813-ac0c24d973b4
# ╠═5d25d703-f49b-4975-baee-d71cf80203fa
# ╟─9bb9ae0d-5149-492d-af64-e33c7f4707fe
# ╠═0ada22b0-fa5f-4d77-a806-96039e72fb10
