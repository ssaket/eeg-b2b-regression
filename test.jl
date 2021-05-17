using PyMNE, DataFrames


function load_eeglab_data(dataPath::String)
    raw = PyMNE.io.read_raw_eeglab(dataPath, preload=true)
    raw.resample(128) # if you want speed ;)

    df = DataFrame(latency=raw.annotations.onset,description=string.(raw.annotations.description),duration=raw.annotations.duration)
    data = raw.get_data()
    return data, df
end

path = "sub-45/eeg/sub-45_task-WLFO_eeg.set"
data, events = load_eeglab_data(path)