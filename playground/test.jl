using PyMNE, DataFrames
using Unfold, StatsModels, Test

# init Plots
using Plots
plotly()
# import custom modules
include("../utils.jl")


# define path for data
path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"

# load data
# data,srate,evts_df,chanlocs_df,EEG = read_raw_eeglab(path)
data, events = read_eeglab(path)
# events[!,:latency] .= events[!,:onset].*128

# """Plots"""
Plots.plot(range(1/50,length=200,step=1/50),data[1:200])
# Plots.vline!(evts_df[evts_df.latency.<=200,:latency]./50) 
# show(first(evts_df,6,),allcols=true)# show events
# gui()
##

#=
define formula
=#
f = @formula 0~1

#= 
Mass Univariate Analysis
=#
data_e_noreshape, times =
    Unfold.epoch(data = data, tbl = events[events.types.=="fixation",:], τ = (-1.0, 1.9), sfreq = 128) # cut the data into epochs
m_mul_noreshape, m_mul_results_noreshape =
    fit(UnfoldLinearModel, f, events, data_e_noreshape, times)


# Special solver solver_lsmr_b2b with b2b regression
se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y)
beta, m_mul_results_b2b =
    Unfold.fit(UnfoldLinearModel, f, events, data_e_noreshape, times, solver = se_solver)