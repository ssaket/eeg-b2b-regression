# eeg-b2b-regression

Back-to-Back regression for encoding and decoding analysis

## Introduction

There are broadly two categories of EEG analyses: Decoding, `f(brain) = stimulus` and encoding `f(stimulus) = brain`. Here, we will try to combine the benefits of both methods based on the analysis-approach of *back2back regression*, which in some sense encompasses both.

## Steps

- We apply back-2-back regression [[`regression.jl`]](./regression.jl) on ground truth data with the help of [Unfold.jl](https://github.com/unfoldtoolbox/unfold.jl)
- We simulate data [[`simulation.jl`]](./simulation.jl), and again apply back-2-back regression to find significant **'face effect'**.
- We use [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook [[`analysis.jl`]](./analysis.jl) to analyse our result.

## Setup

Install dependencies:

 ```julia
julia> # julia REPL
julia> ]  # enter pkg mode
> activate .
> instantiate
 ```

## References

 Ehinger BV, Dimigen O. 2019. Unfold: an integrated toolbox for overlap correction, non-linear modeling, and regression-based EEG analysis. PeerJ 7:e7838 <https://doi.org/10.7717/peerj.7838>

 Jean-Rémi King, François Charton, David Lopez-Paz, Maxime Oquab,
 Back-to-back regression: Disentangling the influence of correlated factors from multivariate observations,
 NeuroImage,
 Volume 220,
 2020,
 117028,
 ISSN 1053-8119,
 <https://doi.org/10.1016/j.neuroimage.2020.117028.(https://www.sciencedirect.com/science/article/pii/S1053811920305140>)
