### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 04ddbcb7-5c2b-4c3b-8cf5-a5b294af8d64
using Unfold, StatsModels

# ╔═╡ 4493bf65-5fec-4be3-bff2-491c2562ce78
include("utils.jl")

# ╔═╡ 369e1538-3855-4d0b-9b17-6f7dd32e9cc9
md"""
# Encoding and Decoding Analysis using Back-2-Back regression 
"""

# ╔═╡ 4cda793b-a766-45de-9fe6-06ffb5a4d847


# ╔═╡ bf87faea-f9b3-4403-867a-422a57034d31
md"""
## Introduction
"""

# ╔═╡ 7f6c9361-dbf2-4fed-88f6-71c2add543aa
md"""
Load events and the EEG data
"""

# ╔═╡ 24e747ea-792c-41d1-9fba-eb7117ccce3a
begin
	
	path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"
	data, events = read_eeglab_with_all_events(path, sfreq=128)
	events = events[events.type .== "fixation",:]
	events[!,:sac_amplitude] = Float64.(events.sac_amplitude)
	events[!,:sac_vmax] = Float64.(events.sac_vmax)
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

# ╔═╡ 0b361dee-8e1e-4ddc-807d-789de6d5e498
md"""
we plot the gamma of Intercept and Saccades amplitude.\
$y = \beta_0 + \beta_1 * sacc\_amp$
"""

# ╔═╡ 5524c3fe-0c6d-48df-80dc-98ddf5172515
begin
	f = @formula 0~1 + sac_amplitude; 
	m, res_exp = get_gamma(f, events, beta, times);
	plot_results(res_exp,layout_x=:basisname);
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
	plot_results(res_vmax,layout_x=:basisname)
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
	plot_results(res_amp_vmax,layout_x=:basisname)
end
	

# ╔═╡ 54a02c16-48f4-4bdc-ba06-3d6af0a971d7
md"""
we plot the gamma of Intercept, human face, saccades amplitude and saccades velocity.\
$y = \beta_0 + \beta_1 * cat(human\_face) + \beta_2 * sac_amplitude + \beta_3 * sac\_vmax$
"""

# ╔═╡ c5dc110c-240c-423e-a660-42644f5c7fcb
begin
	f_amp_vmax_hf = @formula 0~1 + sac_amplitude + sac_vmax + humanface
	_, res_amp_vmax_hf = get_gamma(f_amp_vmax_hf, events, beta, times);
	plot_results(res_amp_vmax_hf,layout_x=:basisname)
end

# ╔═╡ ae0d9783-726d-4b07-a616-c4980935b3f9
md"""

# Playground

"""

# ╔═╡ 311758d8-05a2-48fb-9a32-f0e5cbd40d2a
begin
	fig = Figure(resolution = (800, 600))
	pos = fig[1, 1] = plot_results(res_exp,layout_x=:basisname);
	pos2 = fig[1, 2] = plot_results(res_amp_vmax,layout_x=:basisname)
end

# ╔═╡ 1ca81b86-15fc-498c-8ec0-deef332a720d
begin
	fnew = @formula 0~1 + fix_avgpos_x
	_, res_amp_vmax_pos = get_gamma(fnew, events, beta, times);
	plot_results(res_amp_vmax_pos,layout_x=:basisname)
end

# ╔═╡ 59607be3-3048-4298-9fc2-9535ae6a9837
function get_gamma_ol(path, f)
	p = Progress(3, 1)
	data, events = read_eeglab_with_all_events(path, sfreq=128)
	events = events[events.type .== "fixation",:]
	events[!,:sac_amplitude] = Float64.(events.sac_amplitude)
	events[!,:sac_vmax] = Float64.(events.sac_vmax)
	next!(p)
	
	beta, times = Unfold.epoch(data=data, tbl=events, τ=(-0.3, 0.5), sfreq=128) # cut the data into epochs
	next!(p)
	beta[:,:,isnan.(events.sac_vmax)] .= missing # to run with sac_vmax
	se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y,cross_val_reps = 5)
	next!(p)
	model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)
	next!(p)
	
	return results_expanded
end
	

# ╔═╡ 1701be1f-437d-499c-94cd-6b1a1d15a0a3
begin
	# path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"
	formula = @formula 0~1 + sac_amplitude 
	
	# results_expanded = get_gamma(path, formula)
	plot_results(results_expanded,layout_x=:basisname)
end

# ╔═╡ Cell order:
# ╟─369e1538-3855-4d0b-9b17-6f7dd32e9cc9
# ╠═4cda793b-a766-45de-9fe6-06ffb5a4d847
# ╟─bf87faea-f9b3-4403-867a-422a57034d31
# ╠═04ddbcb7-5c2b-4c3b-8cf5-a5b294af8d64
# ╠═4493bf65-5fec-4be3-bff2-491c2562ce78
# ╟─7f6c9361-dbf2-4fed-88f6-71c2add543aa
# ╠═24e747ea-792c-41d1-9fba-eb7117ccce3a
# ╠═f6cf5ad3-5aad-449b-b563-e7f1a1db7c43
# ╠═cd36c14d-fc59-4327-b807-776f007b1f10
# ╟─c6c154d4-0a74-4fa9-b0ad-5c52b9377c2d
# ╟─0b361dee-8e1e-4ddc-807d-789de6d5e498
# ╠═5524c3fe-0c6d-48df-80dc-98ddf5172515
# ╟─044c89eb-9b08-41ab-a4f6-06dd63f35df2
# ╠═7ad9d3d7-d77b-483b-8e26-bed435345ffd
# ╟─8b79b4e0-8f0f-49ba-b29b-6f5556897847
# ╠═31811918-12b5-4540-b6c7-b18ce88e111d
# ╟─54a02c16-48f4-4bdc-ba06-3d6af0a971d7
# ╠═c5dc110c-240c-423e-a660-42644f5c7fcb
# ╟─ae0d9783-726d-4b07-a616-c4980935b3f9
# ╠═311758d8-05a2-48fb-9a32-f0e5cbd40d2a
# ╠═1ca81b86-15fc-498c-8ec0-deef332a720d
# ╠═59607be3-3048-4298-9fc2-9535ae6a9837
# ╠═1701be1f-437d-499c-94cd-6b1a1d15a0a3
