using DSP, Random
using DataFrames, Distributions
using Unfold, StatsModels

include("utils.jl")
ntime = 100
ntrials = 50
nchannels = 30

function simulate_epoch_data(ntime, ntrials, nchannels; event_ids=undef)
    
    event_names= ["id", "cat_A" ,"cond_A", "cond_B"]

    mv_dist = MvNormal(ones(2), [1 0.5;0.5 1])

    X = rand(mv_dist, ntrials)
    X = hcat(ones(ntrials), rand(0:1, ntrials), X')

    b = DSP.hanning(ntime)
    coef = [0,3,1,2.]
    Yhat = X * (repeat(b', length(event_names)) .* coef)
    
    noiseLevels = Array(1:nchannels)./nchannels
    noise = noiseLevels .* randn(nchannels,ntime,ntrials)

    Yhat = reshape(Yhat', 1, ntime, ntrials)
    Y = repeat(Yhat,nchannels) + noise

    col_names = [Symbol(c) for c in event_names]
    events = DataFrame(X,col_names)

    se_solver = (x, y) -> Unfold.solver_b2b(x, y, cross_val_reps = 5)

    # Generate Designmatrix & fit mass-univariate model (one model per epoched-timepoint) 
    @info "fit mass-univariate model"
    times = Array(1:ntime)
    f = @formula 0~1 + cond_A + cat_A
    model, results_expanded =
        Unfold.fit(UnfoldLinearModel, f, events, Y, times, solver = se_solver)
    plot_results(results_expanded, layout_x = :basisname)
end