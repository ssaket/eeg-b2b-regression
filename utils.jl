using PyMNE, DataFrames, Debugger
using MAT, DelimitedFiles
using AlgebraOfGraphics, GLMakie
using CairoMakie
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
function read_eeglab(filename)
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


function read_eeglab_with_all_events(filename; sfreq::Int64 = 128)
    @bp
    file = MAT.matopen(filename)
    EEG = read(file, "EEG")  # open file
    function parse_struct(s::Dict)
        return DataFrame(map(x -> dropdims(x, dims = 1), values(s)), collect(keys(s)))
    end
    events = parse_struct(EEG["event"])
    @info "Event types"
    println(propertynames(events))

    raw = PyMNE.io.read_raw_eeglab(filename)
    raw.resample(sfreq) # if you want speed ;)
    events[!, :latency] .= raw.annotations.onset .* 128 #add latency

    data = raw.get_data()

    return data, events
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
