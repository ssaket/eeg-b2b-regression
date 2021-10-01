### A Pluto.jl notebook ###
# v0.15.1

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

# ╔═╡ 7225361a-6402-4cca-ad8f-05b7a142eb1d
let
	using Pkg;
	Pkg.activate(".") #pluto doesn't activate the env by default
end

# ╔═╡ f640d374-4b5e-4203-9595-e5df278c1bce
using Unfold, StatsModels, PlutoUI

# ╔═╡ 0dedba95-afef-4532-b2c7-1717942d6fc1
begin
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

# ╔═╡ f44fd85e-ff97-4ba1-9f9a-691a0fb0bb2a
md"""
## Define regularized and un-regularized gamma
"""

# ╔═╡ a5787343-4444-49ac-969c-3b024e5dea4b
function get_gamma(f, events, beta, times)
	se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y,cross_val_reps = 5)
	model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)
	return(model, results_expanded)
end
	

# ╔═╡ c7eee6ca-4d51-4afe-a7cf-aefe93555561
function get_reg_gamma(f, events, beta, times, reg_1, reg_2, reg_3, alpha)
	se_solver = solver = (x, y) -> B2BRegression.solver_b2b(x, y, cross_val_reps = 5, reg_1 = reg_1, reg_2 = reg_2, reg_3 = reg_3, alpha = alpha)
	model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)
	return(model, results_expanded)
end
	

# ╔═╡ 5be5f59b-de28-47a2-a760-2f9e21feaaf5
md"""
Define covariates and their relation
"""

# ╔═╡ 1c5b20c7-f8ff-443b-9e72-5579101c0858
event_ids =
        Dict{Int64,String}(1 => "intercept", 2 => "catA", 3 => "contA", 4 => "contB", 5 => "contC")

# ╔═╡ 89f23206-b095-420e-887b-67ebc53d906b
md"""
### Dynamic events -> change correlation using slider
"""

# ╔═╡ 5dbfc7f4-743a-4add-af4a-a7b062e64730
begin
	sl_times = @bind ntimes PlutoUI.Slider(0:1:600, default=300);
	sl_trials = @bind ntrials PlutoUI.Slider(0:1:1000, default=600);
	sl_channels = @bind nchannels PlutoUI.Slider(1:1:150, default=60);
	
	md""" 
	
	## Define number of trials, channels and times
	
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
begin
	 event_rels_fixed = Dict{String,Union{Vector,Matrix{Float64}}}(
	        "auto_corr" => [],
	        "nominal" => [2],
	        "true_cov" => [1 0.4 0.0; 0.4 1 0.; 0.0 0.0 1],
	    )
	evts = B2BRegression.simulate_events(ntrials, event_ids, event_rels_fixed)
end

# ╔═╡ 648ef3c8-be32-4cbd-97d0-90f21a93451d
sim_data_pink = B2BRegression.simulate_epochs_data(ntimes, nchannels, evts, noise_generator=B2BRegression.pink_noise);

# ╔═╡ 2a3999dc-d648-4337-8420-004c1eb02588
# Pink Noise
function pink_noise(nchannels, ntime, ntrials; max_freq=150, min_freq=30, steps=30)

    freq = range(min_freq, max_freq, length=steps)
    noise = zeros(nchannels, ntime, ntrials)
    c_list = [1,2]
    # sine wave with random phase
    sin_amp = (theta, amp) -> amp*sin(2*pi*theta + 2*rand(1)[1]*pi)

    for ch=1:nchannels
        c = rand(c_list)
        for fi=1:size(freq,1)
            # amplitude = 1/f^c
            amp = 1/freq[fi]^c
            for t=1:ntime
                # summation
                noise[ch,t,:] = noise[ch,t,:] .+ [ sin_amp(freq[fi]*t, amp) for tr in ntrials]
            end
        end
    end
    return noise    
end

# ╔═╡ 52963b4d-629c-4ffa-9a53-67a6f79f56f1
lines(reshape(pink_noise(1, 50, 3), 1,:)[:])

# ╔═╡ f35f29ed-b33a-43ff-821e-54b417557dd7
random_noise(nchannels, ntime, ntrials) = (noiseLevels = Array(1:nchannels) ./ nchannels;
    noiseLevels .* randn(nchannels, ntime, ntrials))

# ╔═╡ 913413d8-2ad6-4e4e-a746-1f5889b6345d
lines(reshape(random_noise(1, 50, 3), 1,:)[:])

# ╔═╡ f06e3a8c-b9f7-4c0c-a7bd-67e6cefbd4c4
begin
	sl = @bind slt PlutoUI.Slider(1:1:ntrials, default=2, show_value=true)
	md"""
	## Simulation Plots
	Trials $(sl)
	"""
	
end

# ╔═╡ 4c5421f8-7dbf-4ef7-9b17-023ef75c38ec
md"""
# Simulation Comparision
"""

# ╔═╡ 448af3c2-6869-4845-88f7-a85deb86efc9
md"""
## Not Regularized
"""

# ╔═╡ 54193936-1063-49ca-af83-bed411fa94b4
md"""
## Regularized
"""

# ╔═╡ 4baa7e56-d6e5-406a-8c74-483469d18b68
md"""
inter 
$(@bind inter PlutoUI.Slider(-2:.1:2,default=1,show_value=true))

correlation_cat_cont (correl1) 
$(@bind corr1 PlutoUI.Slider(-.9:.1:.9,default=0.6,show_value=true))

correlation_cont_cont (correl2) 
$(@bind corr2 PlutoUI.Slider(-.9:.1:.9,default=0.6,show_value=true))

contA (correl)
$(@bind contA PlutoUI.Slider(-2:.1:2,default=1,show_value=true))

contB (uncorr) 
$(@bind contB PlutoUI.Slider(-2:.1:2,default=1,show_value=true))

contC (uncorr) 
$(@bind contC PlutoUI.Slider(-2:.1:2,default=1,show_value=true))

catA 
$(@bind catA PlutoUI.Slider(-2:.1:2,default=1,show_value=true))

alpha (regcoeff)
$(@bind regcoff PlutoUI.Slider(0:.05:5,default=1,show_value=true))
"""



# ╔═╡ a93601a2-1068-4b60-ac8e-3518def6c1f2
 event_rels = Dict{String,Union{Vector,Matrix{Float64}}}(
        "auto_corr" => [],
        "nominal" => [2],
        "true_cov" => [1 corr1 0 0; corr1 1 corr2 0; 0 corr2 1 0; 0 0 0 1],
    )

# ╔═╡ 82eaf5bd-c4c8-4426-acdd-269d4aebcf3f
devts = B2BRegression.simulate_events(ntrials, event_ids, event_rels)

# ╔═╡ 7f386026-c7ee-4535-a55c-dbe518b8653e
sim_data_white = B2BRegression.simulate_epochs_data(ntimes, nchannels, devts, coef=[inter,catA,contA,contB,contC]);

# ╔═╡ 19b03386-84d5-4bb3-a91e-618365ece88c
heatmap(sim_data_white.epochs[:,:,slt])

# ╔═╡ c2559de7-1bc2-4315-943d-e32adfe26a6b
begin
	f_sim_1 = @formula 0~1 + contA + contB
	_, res_sim_1 = get_gamma(f_sim_1, sim_data_white.events, sim_data_white.epochs, sim_data_white.times);
	pt_sim_1 = B2BRegression.plot_results(res_sim_1,layout_x=:basisname)
end

# ╔═╡ 6c5a98e7-81c6-4d37-b5ff-81a966c26002
begin
	f_sim_2 = @formula 0~1 + contA + contB
	_, res_sim_2 = get_gamma(f_sim_2, sim_data_white.events, sim_data_white.epochs, sim_data_white.times);
	pt_sim_2 = B2BRegression.plot_results(res_sim_2,layout_x=:basisname)
end

# ╔═╡ eb2c8bc7-95ea-46fd-a48c-0efa55efeb68
begin
	_low_channel = sim_data_white.epochs[1:3,:,:]
	_high_channel = sim_data_white.epochs[57:60,:,:]
	f_sim_21 = @formula 0~1 + catA + contA + contB + contC
	_, res_sim_21 = get_gamma(f_sim_21, sim_data_white.events, _low_channel, sim_data_white.times);
	pt_sim_21 = B2BRegression.plot_results(res_sim_21,layout_x=:basisname)
end

# ╔═╡ a62d2a6c-533f-4591-b29a-0f2d980eda99
begin
	nsim_data = B2BRegression.simulate_epochs_data(ntimes, nchannels, devts, coef=[inter,catA,contA,contB, contC]);
	
	_, res_0 = get_gamma( @formula(0~1 + catA), nsim_data.events, nsim_data.epochs, nsim_data.times);
	_, res_1 = get_gamma(@formula(0~1 + catA+contA+contB), nsim_data.events, nsim_data.epochs, nsim_data.times);
	
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

# ╔═╡ 46c4c123-c827-4b59-861a-fa002dd22cd7
begin
	rnsim_data = B2BRegression.simulate_epochs_data(ntimes, nchannels, devts, coef=[inter,catA,contA,contB, contC]);
	
	_, rres_0 = get_reg_gamma( @formula(0~1 + catA), rnsim_data.events, rnsim_data.epochs, rnsim_data.times, "l2", "l2", "l0", regcoff);
	_, rres_1 = get_reg_gamma(@formula(0~1 + catA+contA+contB), rnsim_data.events, rnsim_data.epochs, rnsim_data.times, "l2", "l2", "l0", regcoff);
	
	rres_0[!,:group] .= "catAonly"
	rres_1[!,:group] .= "full"
	rres = vcat(rres_0,rres_1)
	rix = rres.term .=="catA"
	rres = rres[ix,:]
	rres[1,:estimate] = 1.
	rres[2,:estimate] = -0.5
# 	# h = plot(res.colname_basis[ix],res.estimate[ix],color="green")
# 	# plot(res.colname_basis[ix],res.estimate[ix],color=res.group)
	rh = B2BRegression.plot_results(rres,layout_x=:basisname, color=:group)
	
	# ylims!(h.grid[1,1].axis,-1.,1)
end

# ╔═╡ Cell order:
# ╟─5c6c3a6a-ebc4-4e15-8b8b-82b8ed7b173b
# ╟─7e41d082-e5a1-11eb-2ab8-e1c33faf9365
# ╟─7225361a-6402-4cca-ad8f-05b7a142eb1d
# ╟─f640d374-4b5e-4203-9595-e5df278c1bce
# ╟─0dedba95-afef-4532-b2c7-1717942d6fc1
# ╟─e0d21513-c591-4c10-82c0-2579a721167b
# ╟─6f7ff757-e79b-437a-b9fa-e1e5238cad90
# ╟─f44fd85e-ff97-4ba1-9f9a-691a0fb0bb2a
# ╟─a5787343-4444-49ac-969c-3b024e5dea4b
# ╟─c7eee6ca-4d51-4afe-a7cf-aefe93555561
# ╟─5be5f59b-de28-47a2-a760-2f9e21feaaf5
# ╟─1c5b20c7-f8ff-443b-9e72-5579101c0858
# ╟─a93601a2-1068-4b60-ac8e-3518def6c1f2
# ╟─89f23206-b095-420e-887b-67ebc53d906b
# ╠═82eaf5bd-c4c8-4426-acdd-269d4aebcf3f
# ╠═5dbfc7f4-743a-4add-af4a-a7b062e64730
# ╟─6092a08d-ea7b-4e81-bda9-ef64d4b52efa
# ╟─49a39dde-8600-47a4-9cc2-2960cc36c963
# ╟─b3a8b686-267a-4c7f-ba89-e2cc04236ace
# ╠═648ef3c8-be32-4cbd-97d0-90f21a93451d
# ╠═7f386026-c7ee-4535-a55c-dbe518b8653e
# ╟─2a3999dc-d648-4337-8420-004c1eb02588
# ╟─52963b4d-629c-4ffa-9a53-67a6f79f56f1
# ╟─f35f29ed-b33a-43ff-821e-54b417557dd7
# ╟─913413d8-2ad6-4e4e-a746-1f5889b6345d
# ╟─f06e3a8c-b9f7-4c0c-a7bd-67e6cefbd4c4
# ╠═19b03386-84d5-4bb3-a91e-618365ece88c
# ╟─c2559de7-1bc2-4315-943d-e32adfe26a6b
# ╟─6c5a98e7-81c6-4d37-b5ff-81a966c26002
# ╠═eb2c8bc7-95ea-46fd-a48c-0efa55efeb68
# ╟─3b70dfc7-0b07-4571-b07b-b3d456eee580
# ╟─4c5421f8-7dbf-4ef7-9b17-023ef75c38ec
# ╟─448af3c2-6869-4845-88f7-a85deb86efc9
# ╟─a62d2a6c-533f-4591-b29a-0f2d980eda99
# ╟─54193936-1063-49ca-af83-bed411fa94b4
# ╟─46c4c123-c827-4b59-861a-fa002dd22cd7
# ╟─4baa7e56-d6e5-406a-8c74-483469d18b68
