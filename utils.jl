using PyMNE, DataFrames
using MAT, DelimitedFiles

# get data from EEGLab using PyMNE 
function read_mne_eeglab(dataPath::String, sfreq::Int64=128)
    raw = PyMNE.io.read_raw_eeglab(dataPath)
    raw.resample(sfreq) # if you want speed ;)
    # get events dataframe
    events = DataFrame(
        latency=raw.annotations.onset .* 128,
        types=string.(raw.annotations.description),
        duration=raw.annotations.duration,
    )
    return raw, events
end

# get data from EEGLab using bare IO
function read_eeglab(filename)
    file = MAT.matopen(filename)
    EEG = read(file, "EEG")  # open file
    function parse_struct(s::Dict)
        return DataFrame(map(x -> dropdims(x, dims=1), values(s)), collect(keys(s)))
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
        data = dropdims(data, dims=3)
    end
    return data, srate, evts_df, chanlocs_df, EEG
end


function read_eeglab_with_all_events(filename; sfreq::Int64=128)
    file = MAT.matopen(filename)
    EEG = read(file, "EEG")  # open file
    function parse_struct(s::Dict)
        return DataFrame(map(x -> dropdims(x, dims=1), values(s)), collect(keys(s)))
    end
    events = parse_struct(EEG["event"])
    raw = PyMNE.io.read_raw_eeglab(filename)

    raw.resample(sfreq) # if you want speed ;)
    data = raw.get_data()
    
    return data, events
end