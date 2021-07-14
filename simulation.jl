using DSP, Random
using DataFrames, Distributions, StatsBase
using Unfold, StatsModels

include("utils.jl")

# Signal windows
hanning_windows(time, padding = 0) = DSP.Windows.hanning(time, padding = padding)
lanczos_windows(time, padding = 0) = DSP.Windows.lanczos(time, padding = padding)
bartlett_hann_windows(time, padding = 0) = DSP.Windows.bartlett_hann(time, padding = padding)

# Mutivariate Distribution
multivariate_ndis(
    means::Vector{Float64} = ones(2),
    covr::Matrix{Float64} = [1 0.5; 0.5 1],
) = MvNormal(means, covr)

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

mutable struct Events
    event_ids::Dict{Int64,String}
    relations::Dict{String,Union{Vector,Matrix{Float64}}}
end

# Generate events data
function simulate_events(
    ntrials::Int64,
    event_ids::Dict{Int64,String},
    relations::Dict{String,Union{Vector,Matrix{Float64}}};
    dist = multivariate_ndis,
    means = ones, # default means are 1
)

    @assert !isempty(relations) "columns relation is required!"
    length(relations["true_cov"]) > 1 &&
        @assert relations["true_cov"] == relations["true_cov"]' "Invalid covariance matrix"

    if length(relations["true_cov"]) < 1
        n = length(event_ids) - 1
        # taken from https://discourse.julialang.org/t/generate-a-positive-definite-matrix/48582
        A = randn(n, n)
        A = A' * A
        A = (A + A') / 2
        relations["auto_corr"] = A
    end

    nom_cols = relations["nominal"]

    # define distribution cov
    true_cov =
        length(relations["true_cov"]) > 1 ? relations["true_cov"] : relations["auto_corr"]
    @assert size(true_cov)[1] == length(event_ids) - 1 "incorrect covariance matrix"
    # define distribution mean
    _mean = means(Float64, size(true_cov)[1])

    X = rand(dist(_mean, true_cov), ntrials)

    # construct events 
    X = hcat(ones(ntrials), X')
    col_names = [Symbol(v) for (_, v) in sort(event_ids)]
    events = DataFrame(X, col_names)

    # normalize and round categorical events
    X = Matrix(events[!, nom_cols])
    dt = fit(UnitRangeTransform, X, dims = 1) # axis = columns
    events[:, nom_cols] .= round.(Int, StatsBase.transform(dt, X))

    return events

end

# Generate epochs data
function simulate_epochs_data(
    ntime,
    nchannels,
    events;
    sampling_rate = 1,
    coef = shuffle(Vector(0:3)), # define channel weights
    noise_generator = random_noise,
    windows = bartlett_hann_windows,
    padding = 0
)
    # signal
    basisfunc = windows(ntime - 2*padding, 2*padding)

    # trials
    ntrials = size(events, 1)

    @assert size(events, 2) == length(coef) "unequal covariates and weights"

    Yhat = Matrix(events) * (repeat(basisfunc', size(events, 2)) .* coef)

    noise = noise_generator(nchannels, ntime, ntrials)

    Yhat = reshape(Yhat', 1, ntime, ntrials)
    beta = repeat(Yhat, nchannels)

    mean_sig, std_sig = mean_and_std(beta)
    mean_noise, std_noise = mean_and_std(noise)
   
    sig_to_noise = (mean_sig - mean_noise) / (std_sig + std_noise)
    @info "Signal to Noise Ratio $(sig_to_noise)"

    beta = beta + noise
    times = range(1, stop = ntime, step = 1 / sampling_rate)

    return SimulationData(events, beta, times)
end

# main function
function run_sim()
    event_ids =
        Dict{Int64,String}(1 => "intercept", 2 => "catA", 3 => "condA", 4 => "condB")
    event_rels = Dict{String,Union{Vector,Matrix{Float64}}}(
        "auto_corr" => [],
        "nominal" => [2],
        "true_cov" => [1 0.7 0.2; 0.7 1 0.3; 0.2 0.3 1],
    )

    events = simulate_events(600, event_ids, event_rels)
    sim_data = simulate_epochs_data(800, 30, events)
    se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

    frm = @formula 0~1 + condA + condB

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

event_ids = Dict{Int64,String}(1 => "intercept", 2 => "catA", 3 => "condA", 4 => "condB")
event_rels = Dict{String,Union{Vector,Matrix{Float64}}}(
    "auto_corr" => [],
    "nominal" => [2],
    "true_cov" => [1 0.5 0.2; 0.5 1 0.3; 0.2 0.3 1],
)
eventsa = simulate_events(50, event_ids, event_rels)
sim_datasa = simulate_epochs_data(100, 30, eventsa)
