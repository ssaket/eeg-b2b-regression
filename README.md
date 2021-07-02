# eeg-b2b-regression
Back-to-Back regression for encoding and decoding analysis

## Introduction 
There are broadly two categories of EEG analyses: Decoding, `f(brain) = stimulus` and encoding `f(stimulus) = brain`. Here, we will try to combine the benefits of both methods based on the analysis-approach of back2back regression, which in some sense encompasses both.

## Steps
 - We apply back-2-back regression(`regression.jl`) on ground truth data with the help of [Unfold.jl](https://github.com/unfoldtoolbox/unfold.jl)
 - We simulate data (`simulation.jl`), and again apply back-2-back regression to find significant effect.
 - We use [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook(`analysis.jl`) to analyse our result.

 ## Setup

Install dependencies:

 ```julia
> ] activate .
> instantiate
 ```