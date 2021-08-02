# imports
using Unfold, StatsModels
include("utils.jl")

# get massunivariate gamma
function get_massunivariate_gamma(
    path::String,
    formula::FormulaTerm;
    τ::Tuple = (-0.3, 0.5),
    event_types::Array{String} = ["fixation"],
    sfreq::Int64 = 128,
    cross_val_reps = 5,
    mixed = false,
)

    @info "reading data and events, with sampling freq $(sfreq)"
    data, events = read_eeglab(path, sfreq)

    @info "selecting only $(event_types) events"
    map(x -> events = events[events.type.==x, :], event_types)

    @info "cutting data into epochs"
    beta, times = Unfold.epoch(data = data, tbl = events, τ = τ, sfreq = sfreq) # cut the data into epochs
    beta[:, :, isnan.(events.sac_vmax)] .= missing # to run with sac_vmax
    se_solver = (x, y) -> solver_b2b(x, y, cross_val_reps = cross_val_reps)

    # Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
    @info "fitting mass-univariate model"
    mtype = mixed ? UnfoldLinearMixedModel : UnfoldLinearModel
    model, results_expanded =
        Unfold.fit(mtype, formula, events, beta, times, solver = se_solver)

    return (model, results_expanded)
end

# plot massunivariate gamma
function plot_massunivariate_gamma(
    path::String,
    formula::FormulaTerm;
    τ::Tuple = (-0.3, 0.5),
    event_types::Array{String} = ["fixation"],
    sfreq::Int64 = 128,
    cross_val_reps = 5,
    mixed = false,
)

    @info "reading data and events, with sampling freq $(sfreq)"
    data, events = read_eeglab(path, sfreq)

    @info "selecting only $(event_types) events"
    map(x -> events = events[events.type.==x, :], event_types)

    @info "cutting data into epochs"
    beta, times = Unfold.epoch(data = data, tbl = events, τ = τ, sfreq = sfreq)
    beta[:, :, isnan.(events.sac_vmax)] .= missing
    se_solver = (x, y) -> solver_b2b(x, y, cross_val_reps = cross_val_reps)

    # Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
    @info "fitting mass-univariate model"
    mtype = mixed ? UnfoldLinearMixedModel : UnfoldLinearModel
    model, results_expanded =
        Unfold.fit(mtype, formula, events, beta, times, solver = se_solver)

    @info "plotting results"
    return plot_results(results_expanded, layout_x = :basisname)
end

# get time-expanded gamma
function get_timeexpansion_gamma(
    path::String,
    eventcolumn,
    config::Dict;
    event_types::Array{String} = ["fixation"],
    sfreq::Int64 = 128,
    channels = [1],
    mixed = false,
)
    @info "reading data and events, with sampling freq $(sfreq)"
    data, events = read_eeglab(path, sfreq)

    @info "selecting only $(event_types) events"
    map(x -> events = events[events.type.==x, :], event_types)

    # select channels
    data = length(channels) > 0 ? reshape(data[channels, :], (1, :)) : reshape(data, (1, :))

    # Generate Designmatrix & fit time-expanded model(modeling linear overlap).
    @info "fitting time-expanded model"
    mtype = mixed ? UnfoldLinearMixedModel : UnfoldLinearModel
    model_new, result_long = fit(mtype, config, events, data, eventcolumn=eventcolumn)
    return (model_new, result_long)
end

# plot time expanded gamma
function plot_timeexpansion_gamma(
    path::String,
    eventcolumn,
    config;
    event_types::Array{String} = ["fixation"],
    sfreq::Int64 = 128,
    channels = [1],
    mixed = false,
)
    @info "reading data and events, with sampling freq $(sfreq)"
    data, events = read_eeglab(path, sfreq)

    @info "selecting only $(event_types) events"
    map(x -> events = events[events.type.==x, :], event_types)

    # select channels
    data = length(channels) > 0 ? reshape(data[channels, :], (1, :)) : reshape(data, (1, :))

    # Generate Designmatrix & fit time-expanded model(modeling linear overlap).
    @info "fitting time-expanded model"
    mtype = mixed ? UnfoldLinearMixedModel : UnfoldLinearModel
    model_new, result_long = fit(mtype, config, events, data, eventcolumn=eventcolumn)

    @info "plotting results"
    return plot_results(result_long, layout_x = :basisname)
end

# main funtion
function start_regression()
    path = "data/sub-45/eeg/sub-45_task-WLFO_eeg.set"
    if !isfile(path)
        path = "/store/data/WLFO/derivatives/preproc_agert/sub-45/eeg/sub-45_task-WLFO_eeg.set"
    end

    f = @formula 0~1 + sac_amplitude + sac_vmax
    plot_massunivariate_gamma(path, f)
    
    # plot_timeexpansion_gamma(
    #     path,
    #     "type",
    #     Dict(
    #         "sac_amplitude" => (
    #             (@formula 0~1 + sac_amplitude + sac_vmax),
    #             firbasis(τ = (-1, 1), sfreq = 128, name = "basisA"),
    #         ),
    #     ),
    # )
end