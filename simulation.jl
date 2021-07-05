using Core: Vector
using DSP, Random
using DataFrames, Distributions, StatsBase
using Unfold, StatsModels

include("utils.jl")

# Signal windows
hanning_windows(time) = DSP.hanning(time)
hamming_windows(time) = DSP.hamming(time)
hanwin_windows(time) = DSP.hanning_winplot(time)

# Mutivariate Distribution
multivariate_ndis(means::Vector{Float64}, covr::Matrix{Float64} = [1 0.5; 0.5 1]) =
    MvNormal(means, covr)

# Noise functions
random_noise(nchannels, ntime, ntrials) = (noiseLevels = Array(1:nchannels) ./ nchannels;
noiseLevels .* randn(nchannels, ntime, ntrials))

# Events mean and covariance
mutable struct EventStatsMatrix
    mean::Matrix{Float64}
    cov::Matrix{Float64}

    EventStatsMatrix(tpl) = new(tpl[1], tpl[2])
end

# Simulation data
mutable struct SimulationData
    data::Array{Float64,2} # channels * times
    events::DataFrame # trials * covariates
    epochs::Array{Float64,3} # channels * times * trials
    times::Array{Float64,1} # trials
    stats::EventStatsMatrix # covariance and mean

    SimulationData(events, epochs, times) = new(
        epochs[:, :, 1],
        events,
        epochs,
        times,
        EventStatsMatrix(mean_and_cov(Matrix(events))),
    )
end


function simulate_events(ntrials, event_ids, relations; dist = multivariate_ndis)

    @assert !isempty(relations) && length(relations["true_cov"]) < 1 "columns relation is required!"

    if !isempty(relations)
        nom_cols = relations["nominal"]
        quan_cols = relations["qualitative"]
        auto_cor_cols = relations["auto_corr"]
    end

    # define distribution cov
    true_cov = [1 0.5; 0.5 1]
    # define distribution mean
    means = ones(Float64, 2)

    X = rand(dist(means, true_cov), ntrials)

    # construct events 
    X = hcat(ones(ntrials), rand(0:1, ntrials), X')
    col_names = [Symbol(v) for (_, v) in sort(event_ids)]
    events = DataFrame(X, col_names)
    return events

end

# Generate simulation data
function simulate_epochs_data(
    ntime,
    ntrials,
    nchannels,
    events,
    sampling_rate = 1;
    noise_generator = random_noise,
    windows = hanning_windows,
)
    b = windows(ntime)
    # define weights
    coef = [0, 3, 1, 2.0]

    @assert length(size(events, 2)) == length(coef)

    Yhat = X * (repeat(b', size(events, 2)) .* coef)

    noise = noise_generator(nchannels, ntime, ntrials)

    Yhat = reshape(Yhat', 1, ntime, ntrials)
    beta = repeat(Yhat, nchannels) + noise
    times = range(1, stop = ntime, step = 1 / sampling_rate)

    return SimulationData(events, beta, times)
end

event_ids = Dict{Int64, String}(
    1 => "intercept",
    2 => "catA",
    3 => "condA",
    4 => "condB",
)
events_relation = Dict{String, Vector{Int64}}(
    "auto_corr" => [],
    "nominal" => [2],
    "qualitative" => [3,4],
    "true_cov" => [1 0.5; 0.5 1]
)

events = simulate_events(ntrials, event_ids, event_rels),
sim_data = simulate_epochs_data(100, 50, 30, events)
se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

frm = @formula 0 ~1 + cond_A + cat_A

# Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
@info "fit mass-univariate model"

model, results_expanded = Unfold.fit(
    UnfoldLinearModel,
    frm,
    sim_data.events,
    sim_data.epochs,
    sim_data.times,
    solver = se_solver,
)
plot_results(results_expanded, layout_x = :basisname)
