using PyMNE, DataFrames
using Unfold, StatsModels, Test

using Plots
gr(size = (300, 300), legend = false)  # provide optional defaults



function load_eeglab_data(dataPath::String)
    raw = PyMNE.io.read_raw_eeglab(dataPath, preload=true)
    raw.resample(128) # if you want speed ;)

    df = DataFrame(latency=raw.annotations.onset,description=string.(raw.annotations.description),duration=raw.annotations.duration)
    data = raw.get_data()
    return data, df
end

include("eegLab.jl")

##
path = "sub-45/eeg/sub-45_task-WLFO_eeg.set"
# data,srate,evts_df,chanlocs_df,EEG = import_eeglab(path)
data, events = load_eeglab_data(path)


# Plots.plot(range(1/50,length=200,step=1/50),data[1:200])
# Plots.vline!(evts_df[evts_df.latency.<=200,:latency]./50) # show events
# gui()
##

# show(first(evts_df,6,),allcols=true)
# data, events = load_eeglab_data(path)
f  = @formula 0~1 + duration

data_e_noreshape,times = Unfold.epoch(data=data,tbl=events,τ=(-1.,1.9),sfreq=20) # cut the data into epochs
# m_mul_noreshape,m_mul_results_noreshape = fit(UnfoldLinearModel,f,events,data_e_noreshape,times)
# @test m_mul_results_noreshape[(m_mul_results_noreshape.channel.==1).&(m_mul_results_noreshape.colname_basis .==0.1),:estimate] ≈ [2,3,4]
# @test size(m_mul_results_noreshape)[1] ==size(m_mul_results)[1]/2


# Special solver solver_lsmr_se with Standard Error
# se_solver = solver=(x,y)->Unfold.solver_default(x,y,stderror=true)
# m_mul_se,m_mul_results_se = Unfold.fit(UnfoldLinearModel,f,events,data_e_noreshape,times,solver=se_solver)
# @test all(m_mul_results_se.estimate .≈ m_mul_results_noreshape.estimate)
# @test !all(isnothing.(m_mul_results_se.stderror ))

# Special solver solver_lsmr_b2b with b2b regression
se_solver = solver=(x,y)->Unfold.solver_b2b(x,y)
beta,m_mul_results_b2b = Unfold.fit(UnfoldLinearModel,f,events,data_e_noreshape,times,solver=se_solver)
