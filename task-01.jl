using Unfold, StatsModels

# init Plots
using Plots
plotly()
# import custom modules
include("utils.jl")


# define path for data
path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"
# raw, events = read_mne_eeglab(path, 128)
# data = raw.get_data()

# Data is typically interpreted as channel x time (with basisfunctions) or channel x time x epoch (for mass univariate)
data, events = read_eeglab_with_all_events(path, sfreq=128)
 
# define formula 
f = @formula 0~1 + duration

# select only fixation events
events = events[events.types .== "fixation",:]

#=  Source: Unfold Documentation
Since, Deconvolution works on continuous data, to compare it to the “normal” use-case, we have to epoch it. 
Data cleaning is doing in the fuction definition 
We additionally remove trials from unfold.X that were removed during epoching 
=#
beta, times = Unfold.epoch(data=data, tbl=events, τ=(-1.0, 1.9), sfreq=128) # cut the data into epochs

# Special solver solver_lsmr_b2b with b2b regression
se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y)

# Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
model, results_expanded = Unfold.fit(UnfoldLinearModel, f, events, beta, times, solver=se_solver)

plot_results(res[res.channel.==1,:],se=true)

# Timexpanded Univariate Linear
b1 = firbasis(τ=(-1,1),sfreq=20,name="basisA")
f1  = @formula 0~1+sac_amplitude

# Generate Designmatrix & fit time-expanded model(modeling linear overlap).
model,results_expanded = Unfold.fit(UnfoldLinearModel,f,evts,data,b1)
model_new, result_long_new = fit(UnfoldLinearModel,Dict(0=>(f1,b1)),evts,data,eventcolumn="fixation")


# TODO
# Add plotting function 