module B2BRegression
# imports
using Unfold, StatsModels
include("utils.jl")

# get massunivariate gamma
function get_massunivariate_gamma(path, formula, event_types = undef)
    @info "reading data and events"
    data, events = read_eeglab_with_all_events(path, sfreq = 128)

    # select only fixation
    @info "selecting only fixation events"
    events = events[events.type.=="fixation", :]
    events[!, :sac_amplitude] = Float64.(events.sac_amplitude)
    events[!, :sac_vmax] = Float64.(events.sac_vmax)

    @info "cutting data into epochs"
    beta, times = Unfold.epoch(data = data, tbl = events, τ = (-0.3, 0.5), sfreq = 128) # cut the data into epochs
    beta[:, :, isnan.(events.sac_vmax)] .= missing # to run with sac_vmax
    se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

    # Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
    @info "fit mass-univariate model"
    model, results_expanded =
        Unfold.fit(UnfoldLinearModel, formula, events, beta, times, solver = se_solver)
    return (model, results_expanded)
end

# plot massunivariate gamma
function plot_massunivariate_gamma(path, formula, event_types = undef, sfreq::Int64=128)

    @info "reading data and events, with sampling freq $(sfreq)"
    data, events = read_eeglab_with_all_events(path; sfreq)

    @info "selecting only fixation events"
    events = events[events.type.=="fixation", :]
    events[!, :sac_amplitude] = Float64.(events.sac_amplitude)
    events[!, :sac_vmax] = Float64.(events.sac_vmax)

    @info "cutting data into epochs"
    beta, times = Unfold.epoch(data = data, tbl = events, τ = (-0.3, 0.5), sfreq = 128) 
    beta[:, :, isnan.(events.sac_vmax)] .= missing
    se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

    # Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
    @info "fit mass-univariate model"
    model, results_expanded =
        Unfold.fit(UnfoldLinearModel, formula, events, beta, times, solver = se_solver)
    @info "plotting results"
    return plot_results(results_expanded, layout_x = :basisname)
end

# get time-expanded gamma
function get_time_expanded_gamma(path, f1, b1, event_types = undef)
    data, events = read_eeglab_with_all_events(path, sfreq = 128)

    # select only fixation
    events = events[events.type.=="fixation", :]
    events[!, :sac_amplitude] = Float64.(events.sac_amplitude)
    events[!, :sac_vmax] = Float64.(events.sac_vmax)

    # Generate Designmatrix & fit time-expanded model(modeling linear overlap).
    # old method
    # model,results_expanded = Unfold.fit(UnfoldLinearModel,f,evts,data,b1) 
    model_new, result_long_new = fit(
        UnfoldLinearModel,
        Dict(0 => (f1, b1)),
        events,
        data,
        eventcolumn = "fixation",
    )
    return (model_new, result_long_new)
end

# plot time expanded gamma
function plot_time_expanded_gamma(path, f1, b1, event_types = undef)
    data, events = read_eeglab_with_all_events(path, sfreq = 128)

    # select only fixation
    events = events[events.type.=="fixation", :]
    events[!, :sac_amplitude] = Float64.(events.sac_amplitude)
    events[!, :sac_vmax] = Float64.(events.sac_vmax)

    # Generate Designmatrix & fit time-expanded model(modeling linear overlap).
    # old method
    model, results_expanded = Unfold.fit(UnfoldLinearModel, f1, events, data, b1)
    # model_new, result_long_new =
    #     fit(UnfoldLinearModel, Dict(0 => (f1, b1)), events, data, eventcolumn = :type, solver = se_solver)
    @info "plotting results"
    return plot_results(results_expanded, layout_x = :basisname)
end

# main funtion
function start_regression()
    path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"
    if !isfile(path)
        path = "/store/data/WLFO/derivatives/preproc_agert/sub-45/eeg/sub-45_task-WLFO_eeg.set"
    end

    f = @formula 0~1 + sac_amplitude + sac_vmax + humanface
    b1 = firbasis(τ = (-1, 1), sfreq = 120, name = "basisA")
    f1 = @formula 0~1 + sac_amplitude + sac_vmax + humanface
    # plot_time_expanded_gamma(path, f1, b1)
    plot_massunivariate_gamma(path, f)
end
end

B2BRegression.start_regression()
