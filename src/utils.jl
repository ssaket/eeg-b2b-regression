using PyMNE, DataFrames, Flux, Unfold
using MAT, DelimitedFiles, MLJLinearModels
using AlgebraOfGraphics, GLMakie
using CairoMakie
using LinearAlgebra
# if, for example, GLMakie is activated already
CairoMakie.activate!()


# get data from EEGLab using PyMNE 
function read_mne_eeglab(dataPath::String, sfreq::Int64 = 128)
    raw = PyMNE.io.read_raw_eeglab(dataPath)
    raw.resample(sfreq) # if you want speed ;)
    # get events dataframe
    events = DataFrame(
        latency = raw.annotations.onset .* 128,
        types = string.(raw.annotations.description),
        duration = raw.annotations.duration,
    )
    return raw, events
end

# get data from EEGLab using bare IO
function read_eeglab_old(filename)
    file = MAT.matopen(filename)
    EEG = read(file, "EEG")  # open file
    function parse_struct(s::Dict)
        return DataFrame(map(x -> dropdims(x, dims = 1), values(s)), collect(keys(s)))
    end
    evts_df = parse_struct(EEG["event"])
    chanlocs_df = parse_struct(EEG["chanlocs"])
    # epoch_df = parse_struct(EEG["epochs"])
    srate = EEG["srate"]
    if typeof(EEG["data"]) == String
        datapath = joinpath(splitdir(filename)[1], EEG["data"])
        data = Array{Float32,3}(
            undef,
            Int(EEG["nbchan"]),
            Int(EEG["pnts"]),
            Int(EEG["trials"]),
        )
        read!(datapath, data)
    else
        data = EEG["data"]
    end
    if (ndims(data) == 3) & (size(data, 3) == 1)
        data = dropdims(data, dims = 3)
    end
    return data, srate, evts_df, chanlocs_df, EEG
end


function read_eeglab(
    filename::String,
    sfreq::Int64;
    cols::Array{String} = String[],
    mne::Bool = false,
    type::Type = Float64,
)

    file = MAT.matopen(filename)
    EEG = read(file, "EEG")  # open file
    evt_cols = EEG["event"]
    # Filtering columns
    length(cols) > 0 && filter!((k, v) -> k in cols, s)
    # Type cast if possible
    function cast_type(x, type)
        try
            return convert.(type, x)
        catch
            return x
        end
    end

    events = DataFrame(
        map(x -> dropdims(x, dims = 1), values(evt_cols)),
        collect(keys(evt_cols)),
    )

    @info "Event types"
    map(x -> events[!, x] = cast_type(events[!, x], type), names(events))
    println(describe(events))

    raw = PyMNE.io.read_raw_eeglab(filename)
    raw.resample(sfreq) # for speed
    events[!, :latency] .= raw.annotations.onset .* sfreq #add latency

    !mne && (raw = raw.get_data())

    return raw, events
end

#---
#Plots.plot(m::Unfold.UnfoldModel)  = plot_results(m.results)

function plot_results(
    results::DataFrame;
    y = :estimate,
    color = :term,
    layout_x = :group,
    stderror = false,
    pvalue = DataFrame(:from => [], :to => [], :pval => []),
)
    m = mapping(:colname_basis, y, color = color, layout_x = layout_x)

    basic = AlgebraOfGraphics.data(results) * visual(Lines) * m

    if stderror
        res_se = copy(results)
        res_se = res_se[.!isnothing.(res_se.stderror), :]
        res_se[!, :se_low] = res_se[:, y] .- res_se.stderror
        res_se[!, :se_high] = res_se[:, y] .+ res_se.stderror
        basic =
            AlgebraOfGraphics.data(res_se) *
            visual(Band, alpha = 0.5) *
            mapping(:colname_basis, :se_low, :se_high, color = :term, layout_x = layout_x) +
            basic
    end

    d = basic |> draw

    # add the pvalues
    if !isempty(pvalue)
        x = [Point(x, 0.0) => Point(y, 0.0) for (x, y) in zip(pvalue.from, pvalue.to)]
        linesegments!(d.children[1], x, linewidth = 2) # assumes first one is where we want to plot. Limitation!

    end
    return d

end

# Custom linear regression function
function linear_default_solver(data, X)

    G = Array{Float64}(undef, size(data, 2), size(X, 2))
    for pred = 1:size(X, 2)
        y = X[:, pred]
        theta = data \ y
        r = y - data * theta
        e = sqrt(sum(abs2.(r)) / size(data, 1))
        # @info "rmse: $(e)"
        G[:, pred] = theta
    end
    return G
end
# Linear regression using MLJLinearModels, produces same output as above function
function linear_solver(data, X)

    linear = LinearRegression(fit_intercept = false)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))

    for pred = 1:size(X, 2)
        y = X[:, pred]
        theta = MLJLinearModels.fit(linear, data, y)
        r = y - data * theta
        e = sqrt(sum(abs2.(r)) / size(data, 1))
        # @info "rmse: $(e)"
        G[:, pred] = theta
    end
    return G
end

# Lasso regularization
function linear_lasso_solver(data, X; lambda = 3.1)

    lasso =
        LassoRegression(lambda = lambda, fit_intercept = false, penalize_intercept = false)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))

    for pred = 1:size(X, 2)
        y = X[:, pred]
        theta = MLJLinearModels.fit(lasso, data, y)
        r = y - data * theta
        e = sqrt(sum(abs2.(r)) / size(data, 1))
        # @info "rmse: $(e)"
        G[:, pred] = theta
    end
    return G
end

# Ridge regularization
function linear_ridge_solver(data, X; lambda = 2.3)
    ridge =
        RidgeRegression(lambda = lambda, fit_intercept = false, penalize_intercept = false)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))

    for pred = 1:size(X, 2)
        y = X[:, pred]
        theta = MLJLinearModels.fit(ridge, data, y)
        r = y - data * theta
        e = sqrt(sum(abs2.(r)) / size(data, 1))
        # @info "rmse: $(e)"
        G[:, pred] = theta
    end
    return G
end

function linear_elastic_solver(data, X; lambda = 2.3, gamma = 1.4)
    elastic = ElasticNetRegression(
        lambda = lambda,
        gamma = gamma,
        fit_intercept = false,
        penalize_intercept = false,
    )
    G = Array{Float64}(undef, size(data, 2), size(X, 2))

    for pred = 1:size(X, 2)
        y = X[:, pred]
        theta = MLJLinearModels.fit(ridge, data, y)
        r = y - data * theta
        e = sqrt(sum(abs2.(r)) / size(data, 1))
        # @info "rmse: $(e)"
        G[:, pred] = theta
    end
    return G
end

map_solver = Dict(
    "l1" => linear_lasso_solver,
    "l2" => linear_ridge_solver,
    "elastic" => linear_elastic_solver,
    "l0" => linear_solver,
    "_" => linear_default_solver, # note to self: remove this after I have agained confidence in MLJLinearModels default linear solver
)

function solver_b2b(
    X,
    data::AbstractArray{T,3};
    cross_val_reps = 5,
    solver = (a, b, c) -> map_solver[c](a, b),
) where {T<:Union{Missing,<:Number}}

    X, data = Unfold.dropMissingEpochs(X, data)
    # standardize the data, important when using regularization
    # https://stats.stackexchange.com/questions/287370/standardization-vs-normalization-for-lasso-ridge-regression
    X[:,2:end] = Flux.normalise(X[:,2:end], dims = 1) # uses StatsBase Z-score to standardize ((X - mean) / sd)
    data = Flux.normalise(data, dims = 1)

    E = zeros(size(data, 2), size(X, 2), size(X, 2))
    W = Array{Float64}(undef, size(data, 2), size(X, 2), size(data, 1))

    prog = Unfold.Progress(size(data, 2) * cross_val_reps, 0.1)
    for t = 1:size(data, 2)

        for m = 1:cross_val_reps
            k_ix = collect(Unfold.Kfold(size(data, 3), 2))
            Y1 = data[:, t, k_ix[1]]'
            Y2 = data[:, t, k_ix[2]]'
            X1 = X[k_ix[1], :]
            X2 = X[k_ix[2], :]


            G = solver(Y1, X1, "l1")
            H = solver(X2, (Y2 * G), "l1")

            E[t, :, :] = E[t, :, :] + Diagonal(H[diagind(H)])
            Unfold.ProgressMeter.next!(prog; showvalues = [(:time, t), (:cross_val_rep, m)])
        end
        E[t, :, :] = E[t, :, :] ./ cross_val_reps
        W[t, :, :] = solver((X * E[t, :, :]), data[:, t, :]', "l0")

    end

    # extract diagonal
    beta = mapslices(diag, E, dims = [2, 3])
    # reshape to conform to ch x time x pred
    beta = permutedims(beta, [3 1 2])
    modelinfo = Dict("W" => W, "E" => E, "cross_val_reps" => cross_val_reps) # no history implemented (yet?)
    return beta, modelinfo
end