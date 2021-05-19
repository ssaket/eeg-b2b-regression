##
using StatsModels, MixedModels, DataFrames
import DSP.conv
import Plots
using Unfold
using Test

include("../Unfold.jl/test/test_utilities.jl"); # to load the simulated data

data,evts = loadtestdata("test_case_3a") #
f  = @formula 0~1+conditionA+continuousA # 1

# prepare data
data_r = reshape(data,(1,:))
data_r = vcat(data_r,data_r)#add second channel

#--------------------------#
## Mass Univariate Linear ##
#--------------------------#
data_e,times = Unfold.epoch(data=data_r,tbl=evts,τ=(-1.,1.9),sfreq=20) # cut the data into epochs
m_mul,m_mul_results = fit(UnfoldLinearModel,f,evts,data_e,times)
@test m_mul_results[(m_mul_results.channel.==1).&(m_mul_results.colname_basis .==0.1),:estimate] ≈ [2,3,4]


data_e_noreshape,times = Unfold.epoch(data=data,tbl=evts,τ=(-1.,1.9),sfreq=20) # cut the data into epochs
m_mul_noreshape,m_mul_results_noreshape = fit(UnfoldLinearModel,f,evts,data_e_noreshape,times)
@test m_mul_results_noreshape[(m_mul_results_noreshape.channel.==1).&(m_mul_results_noreshape.colname_basis .==0.1),:estimate] ≈ [2,3,4]
@test size(m_mul_results_noreshape)[1] ==size(m_mul_results)[1]/2

# Add Missing in Data
data_e_missing = data_e
data_e_missing[1,25,end-5:end] .= missing
m_mul_missing,m_mul_missing_results = Unfold.fit(UnfoldLinearModel,f,evts,data_e_missing,times)
@test m_mul_missing_results.estimate ≈ m_mul_results.estimate

# Special solver solver_lsmr_se with Standard Error
se_solver = solver=(x,y)->Unfold.solver_b2b(x,y)
m_mul_se,m_mul_results_se = Unfold.fit(UnfoldLinearModel,f,evts,data_e,times,solver=se_solver)
@test all(m_mul_results_se.estimate .≈ m_mul_results.estimate)
@test !all(isnothing.(m_mul_results_se.stderror ))

