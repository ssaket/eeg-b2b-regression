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


function simulate_events(
    ntrials::Int64,
    event_ids::Dict{Int64, String},
    relations::Dict{String, Union{Vector, Matrix{Float64}}};
    dist = multivariate_ndis,
    means = ones,
)

    @assert !isempty(relations) "columns relation is required!"
    @assert relations["true_cov"] == relations["true_cov"]' "Invalid covariance matrix"

    if length(relations["true_cov"]) < 1
        n = length(event_ids)
        # taken from https://discourse.julialang.org/t/generate-a-positive-definite-matrix/48582
        relations["auto_corr"] = randn(n,n); A = A'*A; A = (A + A')/2
    end

    nom_cols = relations["nominal"]

    # define distribution cov
    true_cov = length(relations["true_cov"]) > 1 ? relations["true_cov"] : relations["auto_corr"]
    @assert size(true_cov)[1] == length(event_ids) - 1 "incorrect covariance matrix"
    # define distribution mean
    _mean = means(Float64, size(true_cov)[1])

    X = rand(dist(_mean, true_cov), ntrials)

    # construct events 
    X = hcat(ones(ntrials), X')
    col_names = [Symbol(v) for (_, v) in sort(event_ids)]
    events = DataFrame(X, col_names)
    events[:, nom_cols] .= round.(Int, events[!, nom_cols]) 
    return events

end

# Generate simulation data
function simulate_epochs_data(
    ntime,
    nchannels,
    events,
    sampling_rate = 1;
    noise_generator = random_noise,
    windows = hanning_windows,
)
    # signal
    b = windows(ntime)
    # define weights
    coef = [0, 3, 1, 2.0]
    # trials
    ntrials = size(events, 1)

    @assert size(events, 2) == length(coef) "unequal covariates and weights"

    Yhat = Matrix(events) * (repeat(b', size(events, 2)) .* coef)

    noise = noise_generator(nchannels, ntime, ntrials)

    Yhat = reshape(Yhat', 1, ntime, ntrials)
    beta = repeat(Yhat, nchannels) + noise
    times = range(1, stop = ntime, step = 1 / sampling_rate)

    return SimulationData(events, beta, times)
end


function run_sim()
    event_ids = Dict{Int64,String}(1 => "intercept", 2 => "catA", 3 => "condA", 4 => "condB")
    event_rels = Dict{String, Union{Vector, Matrix{Float64}}}(
        "auto_corr" => [],
        "nominal" => [2],
        "qualitative" => [3, 4],
        "true_cov" => [1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1]
    )

    events = simulate_events(50, event_ids, event_rels)
    sim_data = simulate_epochs_data(100, 30, events)
    se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

    frm = @formula 0~1 + condA + catA

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
end

run_sim()
