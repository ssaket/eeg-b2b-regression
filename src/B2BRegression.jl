module B2BRegression

export get_massunivariate_gamma,
    plot_massunivariate_gamma,
    get_timeexpansion_gamma,
    plot_timeexpansion_gamma,
    simulate_events,
    simulate_epochs_data,
    run_simulation,
    plot_results,
    read_eeglab

include("regression.jl")
include("simulation.jl")

end #module