using PyMNE, DataFrames
using Unfold, StatsModels, Test

# init Plots
using Plots
gr()
# import custom modules
include("../utils.jl")


# define path for data
path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"

# load data
# data,srate,evts_df,chanlocs_df,EEG = import_eeglab(path)
data, events = read_eeglab(path)

# """Plots"""
Plots.plot(range(1/50,length=200,step=1/50),data[1:200])
# Plots.vline!(evts_df[evts_df.latency.<=200,:latency]./50) 
# show(first(evts_df,6,),allcols=true)# show events
# gui()
##

#=
define formula
=#
f = @formula 0~1 + duration

#= 
Mass Univariate Analysis
=#
data_e_noreshape, times =
    Unfold.epoch(data = data, tbl = events, τ = (-1.0, 1.9), sfreq = 20) # cut the data into epochs
m_mul_noreshape, m_mul_results_noreshape =
    fit(UnfoldLinearModel, f, events, data_e_noreshape, times)
# @test m_mul_results_noreshape[(m_mul_results_noreshape.channel.==1).&(m_mul_results_noreshape.colname_basis .==0.1),:estimate] ≈ [2,3,4]
# @test size(m_mul_results_noreshape)[1] ==size(m_mul_results)[1]/2


# Special solver solver_lsmr_se with Standard Error
# se_solver = solver=(x,y)->Unfold.solver_default(x,y,stderror=true)
# m_mul_se,m_mul_results_se = Unfold.fit(UnfoldLinearModel,f,events,data_e_noreshape,times,solver=se_solver)
# @test all(m_mul_results_se.estimate .≈ m_mul_results_noreshape.estimate)
# @test !all(isnothing.(m_mul_results_se.stderror ))
# @test all(m_mul_results_se.estimate .≈ m_mul_results_noreshape.estimate)
# @test !all(isnothing.(m_mul_results_se.stderror ))

# Special solver solver_lsmr_b2b with b2b regression
se_solver = solver = (x, y) -> Unfold.solver_b2b(x, y)
beta, m_mul_results_b2b =
    Unfold.fit(UnfoldLinearModel, f, events, data_e_noreshape, times, solver = se_solver)