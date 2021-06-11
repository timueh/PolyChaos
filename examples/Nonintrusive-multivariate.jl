## This is an example on the usage of PolyChaos for non-intrusive PCE using
## the projection and the regression approach and a comarison of them in a multi-
## variate setting.
## Model: f = ΣX^2, X = {X_1, ..., X_k}, X_i ~ N(0, 1), Y ~ Χ²(k)

using PolyChaos
using LinearAlgebra
using Distributions


### Parameters ###
μ, σ = 0, 1
maxdeg = 2
Nrec = 20
k = 12 # degrees of freedom

### Model equation ###
function model(x)
    x.^2
end

### Setup and compute PCE coefficients of x ###
op = GaussOrthoPoly(maxdeg, Nrec=Nrec, addQuadrature=true)
mop = MultiOrthoPoly([op for i in 1:k], maxdeg)

# Find PCE for all X_i
L = dim(mop)
# assign2multi(): take coefficients x for univariate elements and assign then to their accoring multi index. 
# If x has 1 element, only take degree 1 basis elements, if it has 2 take the first 2 etc.
x = [ assign2multi(convert2affinePCE(μ, σ, op), i, mop.ind) for i in 1:k ]
# we can describe a gaussian RV with ~N(0,1) by PCE coefficients x_0 = 1 and x_1 = 1

# Compute tensors <ϕ_m,ϕ_m> and <ϕ_1,ϕ_2,ϕ_m>
t2 = Tensor(2, mop)
t3 = Tensor(3, mop)


# ------- Reference -------
### Intrusive PCE ###
println("\t == Galerking Projection ==")
# x[i][j] - i = which variable, j = coefficient for degree j
y_intr = [ sum( x[i][j1] * x[i][j2] * t3.get([j1-1, j2-1, m-1]) for i = 1:k, j1 = 1:L, j2 = 1:L ) / t2.get([m-1, m-1]) for m in 1:L ]
println(y_intr)
    
### Analytic ###
# ...


# ------- Projection -------
# Non-intrusive part starts here
# println("\t == Projection approach ==")

# Perform projection via numerical integration
# function computeProjection(k::Int, model::Function)
    # g(t) = model(t) .* evaluate(k, t, op)
    # γ = t2.get([k,k])
    # integrate(g, op.quad) / γ
# end

# y_proj = computeProjection.(0:maxdegree, model)
# println.(y_proj)
# println()

# # Comparison to analytical solution
# println("Comapre coefficients analytic <-> projection:")
# println(norm(y_ana - y_proj, Inf), "\n")



# ------- Regression -------
include("../src/regression.jl")
println("\t == Regression approach ==")

# FUTURE: Perform truncation somehow

# Draw n samples, where N > P has to hold
# nSamples = maxdeg * 20
# X = sampleMeasure(nSamples, mop)

# Evaluate model -> vector Y
# Y = model.(X)

# Build matrix Φ with ϕ(x(i))
# Φ = Array{Float64}(undef, nSamples, maxdegree+1)
# Φ = [ evaluate(j, X[i], op) for i = 1:nSamples, j = 0:maxdeg]

# Ordinary least squares regression
# y_reg = leastSquares(Φ, Y)
# println.(y_reg)
# println()


# Validation of PCE model
# println("Comapre coefficients analytic <-> regression:")
# println(norm(y_ana - y_reg, Inf))

# genError = empError(Y, Φ, y_reg)
# println("Determination coefficient R² (normalized empicial error): ", 1 - genError)

# ϵLoo = looError(Y, Φ, y_reg)
# println("Determination coefficient Q² (leave-one-out error): ", 1- ϵLoo)
# println()



# ------- Monte-Carlo -------
# Evaluate model function on same set of previously drawn samples X
# y_mc = model.(X)



# ------- Comparison of moments -------
println("\nComparison of moments to analytic solution")

# Analytic moments for y
mean_ana = k
std_ana = sqrt(2*k)
skew_ana = sqrt(8/k)

# PCE skewness
function skew(y)
    e3 = sum( y[i] * y[j] * y[k] * t3.get([i-1, j-1, k-1]) for i = 1:L, j = 1:L, k = 1:L )
    μ = y[1]
    σ = std(y, mop)
    (e3 - 3 * μ * σ^2 - μ^3) / (σ^3)
end

# Intrusive PCE moments
mean_intr = mean(y_intr, mop)
std_intr  = std(y_intr, mop)
skew_intr = skew(y_intr)
error_mean_intr = abs(mean_ana - mean_intr)
error_std_intr  = abs(std_ana - std_intr)
error_skew_intr = abs(skew_ana - skew_intr)
println("= Error Intrusive PCE vs Analytic")
println("\t\t\t error mean: \t $(error_mean_intr)")
println("\t\t\t error std: \t $(error_std_intr)")
println("\t\t\t error skew: \t $(error_skew_intr)\n")


# # MC moments
# mean_mc, std_mc, skew_mc = mean(y_mc), std(y_mc), skewness(y_mc)
# error_mean_mc = abs(mean_ana - mean_mc)
# error_std_mc = abs(std_ana - std_mc)
# println("\t\t\t error MC, mean: \t $(error_mean_mc)")
# println("\t\t\t error MC, std: \t $(error_std_mc)")

# # Projection moments
# mean_proj = mean(y_proj, op)
# std_proj = std(y_proj, op)
# error_mean_proj = abs(mean_ana - mean_proj)
# error_std_proj = abs(std_ana - std_proj)
# println("\t\t\t error proj, mean: \t $(error_mean_proj)")
# println("\t\t\t error proj, std: \t $(error_std_proj)")

# # Regression moments
# mean_reg = mean(y_reg, op)
# std_reg = std(y_reg, op)
# error_mean_reg = abs(mean_ana - mean_reg)
# error_std_reg = abs(std_ana - std_reg)
# println("\t\t\t error reg, mean: \t $(error_mean_reg)")
# println("\t\t\t error reg, std: \t $(error_std_reg)")