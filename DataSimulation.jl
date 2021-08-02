### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f640d374-4b5e-4203-9595-e5df278c1bce
using Unfold, StatsModels

# ╔═╡ 0dedba95-afef-4532-b2c7-1717942d6fc1
begin
	import PlutoUI
	using CairoMakie
	CairoMakie.activate!()
end

# ╔═╡ 6f7ff757-e79b-437a-b9fa-e1e5238cad90
begin
	include("src/B2BRegression.jl");
	using .B2BRegression;
end

# ╔═╡ 5c6c3a6a-ebc4-4e15-8b8b-82b8ed7b173b
md"""
# Encoding and Decoding Analysis using Back-to-Back regression
"""

# ╔═╡ 7e41d082-e5a1-11eb-2ab8-e1c33faf9365
md"""
## Data Simulation
"""

# ╔═╡ e0d21513-c591-4c10-82c0-2579a721167b
md"""
Load B2BRegression Module
"""

# ╔═╡ a5787343-4444-49ac-969c-3b024e5dea4b
function get_gamma(f, events, beta, times)
	se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y,cross_val_reps = 5)
	model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)
	return(model, results_expanded)
end
	

# ╔═╡ 5be5f59b-de28-47a2-a760-2f9e21feaaf5
md"""
Define covariates and their relation
"""

# ╔═╡ 1c5b20c7-f8ff-443b-9e72-5579101c0858
event_ids =
        Dict{Int64,String}(1 => "intercept", 2 => "catA", 3 => "condA", 4 => "condB")

# ╔═╡ a93601a2-1068-4b60-ac8e-3518def6c1f2
 event_rels = Dict{String,Union{Vector,Matrix{Float64}}}(
        "auto_corr" => [],
        "nominal" => [2],
        "true_cov" => [1 0.3 0.0; 0.3 1 0.; 0.0 0.0 1],
    )

# ╔═╡ 5dbfc7f4-743a-4add-af4a-a7b062e64730
begin
	sl_times = @bind ntimes PlutoUI.Slider(0:5:600, default=200);
	sl_trials = @bind ntrials PlutoUI.Slider(0:5:600, default=100);
	sl_channels = @bind nchannels PlutoUI.Slider(1:1:40, default=30);
	
	md""" **Define number of trials, channels and times**
	
	Times: $(sl_times)
	
	Trials: $(sl_trials)
	
	Channels: $(sl_channels)
	
	"""
end

# ╔═╡ 6092a08d-ea7b-4e81-bda9-ef64d4b52efa
md""" 
**Running for**\
Times $(ntimes), Trials $(ntrials), Channels $(nchannels)
"""

# ╔═╡ 49a39dde-8600-47a4-9cc2-2960cc36c963
md"""
Events table
"""

# ╔═╡ b3a8b686-267a-4c7f-ba89-e2cc04236ace
evts = B2BRegression.simulate_events(ntrials, event_ids, event_rels)

# ╔═╡ 648ef3c8-be32-4cbd-97d0-90f21a93451d
sim_data = B2BRegression.simulate_epochs_data(ntimes, nchannels, evts)

# ╔═╡ f06e3a8c-b9f7-4c0c-a7bd-67e6cefbd4c4
begin
	sl = @bind slt PlutoUI.Slider(1:1:ntrials, default=2, show_value=true)
	md"""
	## Simulation Plots
	Trials $(sl)
	"""
	
end

# ╔═╡ 5ef21ab4-dcee-4273-8bf6-5faaea31f612
heatmap(sim_data.epochs[:,:,slt]')

# ╔═╡ 3f1c2145-60c5-4127-b906-109f997679fc
B2BRegression.run_simulation(10)

# ╔═╡ 4baa7e56-d6e5-406a-8c74-483469d18b68
begin	
	md"""
	inter 
	$(@bind inter PlutoUI.Slider(-2:.1:2,default=1,show_value=true))
	
	correlation (correl) 
	$(@bind corr PlutoUI.Slider(-2:.1:2,default=1,show_value=true))
	
	condA (correl)
	$(@bind condA PlutoUI.Slider(-2:.1:2,default=1,show_value=true))
	
	condB (uncorr) 
	$(@bind condB PlutoUI.Slider(-2:.1:2,default=1,show_value=true))
	
	catA 
	$(@bind catA PlutoUI.Slider(-2:.1:2,default=1,show_value=true))
	"""
end

# ╔═╡ c2559de7-1bc2-4315-943d-e32adfe26a6b
begin
	f_sim_1 = @formula 0~1 + condA + condB
	_, res_sim_1 = get_gamma(f_sim_1, sim_data.events, sim_data.epochs, sim_data.times);
	pt_sim_1 = plot_results(res_sim_1,layout_x=:basisname)
end

# ╔═╡ 6c5a98e7-81c6-4d37-b5ff-81a966c26002
begin
	f_sim_2 = @formula 0~1 + catA + condB
	_, res_sim_2 = get_gamma(f_sim_2, sim_data.events, sim_data.epochs, sim_data.times);
	pt_sim_2 = plot_results(res_sim_2,layout_x=:basisname)
end

# ╔═╡ a62d2a6c-533f-4591-b29a-0f2d980eda99
begin
	nsim_data = B2BRegression.simulate_epochs_data(ntimes, nchannels, evts, coef=[inter,catA,condA,condB]);
	
	_, res_0 = get_gamma( @formula(0~1 + catA), nsim_data.events, nsim_data.epochs, sim_data.times);
	_, res_1 = get_gamma(@formula(0~1 + catA+condA+condB), nsim_data.events, nsim_data.epochs, nsim_data.times);
	
	res_0[!,:group] .= "catAonly"
	res_1[!,:group] .= "full"
	res = vcat(res_0,res_1)
	ix = res.term .=="catA"
	res = res[ix,:]
	res[1,:estimate] = 1.
	res[2,:estimate] = -0.5
	# h = plot(res.colname_basis[ix],res.estimate[ix],color="green")
	# plot(res.colname_basis[ix],res.estimate[ix],color=res.group)
	h = B2BRegression.plot_results(res,layout_x=:basisname, color=:group)
	
	# ylims!(h.grid[1,1].axis,-1.,1)
end

# ╔═╡ 3b70dfc7-0b07-4571-b07b-b3d456eee580
res

# ╔═╡ Cell order:
# ╟─5c6c3a6a-ebc4-4e15-8b8b-82b8ed7b173b
# ╟─7e41d082-e5a1-11eb-2ab8-e1c33faf9365
# ╠═f640d374-4b5e-4203-9595-e5df278c1bce
# ╠═0dedba95-afef-4532-b2c7-1717942d6fc1
# ╟─e0d21513-c591-4c10-82c0-2579a721167b
# ╠═6f7ff757-e79b-437a-b9fa-e1e5238cad90
# ╠═a5787343-4444-49ac-969c-3b024e5dea4b
# ╟─5be5f59b-de28-47a2-a760-2f9e21feaaf5
# ╠═1c5b20c7-f8ff-443b-9e72-5579101c0858
# ╠═a93601a2-1068-4b60-ac8e-3518def6c1f2
# ╟─5dbfc7f4-743a-4add-af4a-a7b062e64730
# ╟─6092a08d-ea7b-4e81-bda9-ef64d4b52efa
# ╟─49a39dde-8600-47a4-9cc2-2960cc36c963
# ╠═b3a8b686-267a-4c7f-ba89-e2cc04236ace
# ╠═648ef3c8-be32-4cbd-97d0-90f21a93451d
# ╟─f06e3a8c-b9f7-4c0c-a7bd-67e6cefbd4c4
# ╠═5ef21ab4-dcee-4273-8bf6-5faaea31f612
# ╠═c2559de7-1bc2-4315-943d-e32adfe26a6b
# ╠═6c5a98e7-81c6-4d37-b5ff-81a966c26002
# ╠═3f1c2145-60c5-4127-b906-109f997679fc
# ╟─4baa7e56-d6e5-406a-8c74-483469d18b68
# ╠═3b70dfc7-0b07-4571-b07b-b3d456eee580
# ╠═a62d2a6c-533f-4591-b29a-0f2d980eda99
