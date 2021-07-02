using Unfold, StatsModels

# init Plots
# using Plots # I dont use Plots but Makie.jl
# plotly()
include("utils.jl")


# define dataset path
path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"
if !isfile(path)
    path = "/store/data/WLFO/derivatives/preproc_agert/sub-45/eeg/sub-45_task-WLFO_eeg.set"
end
# raw, events = read_mne_eeglab(path, 128)
# data = raw.get_data()

# Data is typically interpreted as channel x time (with basisfunctions) or channel x time x epoch (for mass univariate)
data, events = read_eeglab_with_all_events(path, sfreq=128)
 
# define formula 
f = @formula 0~1 +  sac_amplitude # also tried + sac_vmax with interesting results


# select only fixation
events = events[events.type .== "fixation",:]
events[!,:sac_amplitude] = Float64.(events.sac_amplitude)
events[!,:sac_vmax] = Float64.(events.sac_vmax)



#= Source: Unfold Documentation
Since, Deconvolution works on continuous data, to compare it to the “normal” use-case, we have to epoch it. 
Data cleaning is doing in the fuction definition 
We additionally remove trials from unfold.X that were removed during epoching 
=#
beta, times = Unfold.epoch(data=data, tbl=events, τ=(-0.3, 0.5), sfreq=128) # cut the data into epochs

beta[:,:,isnan.(events.sac_vmax)] .= missing # to run with sac_vmax

# Special solver solver_lsmr_b2b with b2b regression

# depending on unfold version, you might not be able to use the named argument, but have to put solver_b2b(x,y,1)
se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y,cross_val_reps = 5)


# Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)

# include("dev/Unfold/src/plot.jl") # install Makie first, also run ]dev --local Unfold   to get a "local copy" of Unfold in the folder ./dev/Unfold

plot_results(results_expanded,layout_x=:basisname) # the layout_x is needed because group=nothing and the plotting function doesnt like that...

#--- didnt look further than that, but noticed I havent implemented B2B for 2D data yet (necessary for deconvolution) - but it should be very simple to do.

# select a channel? Is it necessary?plo
# results = results_expanded[results_expanded.channel.==1,:]
# plot(x = results[:colname_basis], y = results[:estimate])

# # Timexpanded Univariate Linear
# b1 = firbasis(τ=(-1,1),sfreq=20,name="basisA")
# f1  = @formula 0~1+sac_amplitude

# # Generate Designmatrix & fit time-expanded model(modeling linear overlap).
# model,results_expanded = Unfold.fit(UnfoldLinearModel,f,evts,data,b1)
# model_new, result_long_new = fit(UnfoldLinearModel,Dict(0=>(f1,b1)),evts,data,eventcolumn="fixation")


# TODO
# Add plotting function 