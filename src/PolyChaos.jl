__precompile__()

module PolyChaos

using SpecialFunctions, SparseArrays, Distributions
import LinearAlgebra: I, dot, SymTridiagonal, eigen, issymmetric, pinv, cond
import FFTW: ifft
import Combinatorics: with_replacement_combinations
import Base: show
import AdaptiveRejectionSampling: RejectionSampler, run_sampler!
import Statistics: mean, std, var
import GaussQuadrature: special_eigenproblem!

include("typesAbstract.jl")
include("typesMeasures.jl")
include("typesQuad.jl")
include("typesOrthoPolys.jl")
include("typesTensor.jl")
include("multi_indices.jl")
include("evaluate.jl")
include("quadrature_rules.jl")
include("recurrence_coefficients_monic.jl")
include("stieltjes.jl")
include("densities.jl")
include("auxfuns.jl")
include("scalar_product.jl")
include("tensor.jl")
include("show.jl")
include("polynomial_chaos.jl")

# Currently under development
include("regression.jl")
include("error_estimation.jl")
# include("adaptive_basis.jl")


end
