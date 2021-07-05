using DSP, Random
using DataFrames, Distributions
using Unfold, StatsModels

include("utils.jl")

hanning_windows(time) = DSP.hanning(time)
hamming_windows(time) = DSP.hamming(time)
hanwin_windows(time) = DSP.hanning_winplot(time)

random_noise(nchannels, ntime, ntrials) = (noiseLevels = Array(1:nchannels) ./ nchannels;
noiseLevels .* randn(nchannels, ntime, ntrials))

frm = @formula 0~1 + cond_A + cat_A

event_ids = Dict(1 => "id", 2 => "cat_A", 3 => "cond_A", 4 => "cond_B")

struct EpochData
    events::DataFrames
    data::Array
    times::Array

    EpochData(events, data, times) = new(events, data, times)
end


function simulate_epoch_data(
    ntime = 100,
    ntrials = 50,
    nchannels = 30;
    event_ids = event_ids;
    noise_generator = random_noise;
    windows = hanning_windows
)
    mv_dist = MvNormal(ones(2), [1 0.5; 0.5 1])

    X = rand(mv_dist, ntrials)
    X = hcat(ones(ntrials), rand(0:1, ntrials), X')
    col_names = [Symbol(v) for (k, v) in event_ids]
    events = DataFrame(X, col_names)


    b = windows(ntime)
    coef = [0, 3, 1, 2.0]

    @assert length(event_ids) == length(coef)

    Yhat = X * (repeat(b', length(event_ids)) .* coef)

    noise = noise_generator(nchannels, ntime, ntrials)

    Yhat = reshape(Yhat', 1, ntime, ntrials)
    beta = repeat(Yhat, nchannels) + noise
    times = Array(1:ntime)

    epoch_data = EpochData(events, beta, times)

    return epoch_data
end


se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

# Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
@info "fit mass-univariate model"

model, results_expanded =
    Unfold.fit(UnfoldLinearModel, f, events, Y, times, solver = se_solver)
plot_results(results_expanded, layout_x = :basisname)
