# eeg-b2b-regression

Back-to-Back regression for encoding and decoding analysis

## Introduction

There are broadly two categories of EEG analyses: Decoding, `f(brain) = stimulus` and encoding `f(stimulus) = brain`. Here, we will try to combine the benefits of both methods based on the analysis-approach of *back2back regression*, which in some sense encompasses both.

## Steps

We use [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks to analyse our result.

- We simulate EEG data [[`DataSimulation.jl`]](./DataSimulation.jl), and apply back-2-back regression using regularized(L1, L2, Elastic) and un-regularized solvers with the help of [Unfold.jl](https://github.com/unfoldtoolbox/unfold.jl). We also explore single layer neural network solver.

- We apply back-2-back regression [[`DataAnalysis.jl`]](./DataAnalysis.jl) on ground truth data to disentangle the effects of continuous correlated predictors and uncorrelated categorical predictor.

- We generate [saliency maps](./saliency/README.md) and analyse saliency scores.

## Setup

Install dependencies:

 ```julia
julia> # julia REPL
julia> ]  # enter pkg mode
> activate .
> instantiate
 ```

 To run Pluto
 ```julia
julia> import Pluto;
julia> Pluto.run()
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
