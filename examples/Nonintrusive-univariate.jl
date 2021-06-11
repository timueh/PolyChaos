using PolyChaos
using LinearAlgebra
using Distributions


## This is an example on the usage of PolyChaos for non-intrusive PCE using
## the projection and the regression approach and a comarison of them in the univariate setting.
## f = e^x, x ~ N(1, 0.1)

μ, σ = 1, 0.1
maxdegree = 7

# model equation
function model(x)
    exp.(μ + σ * x)
end


# Analytic refernce
k_values = 0:maxdegree
computeCoefficient_analytic(k::Int) = exp(μ + 0.5*σ^2) * σ^k / factorial(k)
y_ana = computeCoefficient_analytic.(k_values)


# Setup and compute PCE coefficients of x
op = GaussOrthoPoly(maxdegree, Nrec=2*maxdegree)
t2 = Tensor(2, op)



# ------- Projection -------
println("\t == Projection approach ==")

# Perform projection via numerical integration
function computeProjection(k::Int, model::Function)
    g(t) = model(t) .* evaluate(k, t, op)
    γ = t2.get([k,k])
    integrate(g, op.quad) / γ
end

y_proj = computeProjection.(0:maxdegree, model)
println.(y_proj)
println()

# Comparison to analytical solution
println("Comapre coefficients analytic <-> projection:")
println(norm(y_ana - y_proj, Inf), "\n")



# ------- Regression -------
include("../src/regression.jl")

println("\t == Regression approach ==")

# FUTURE: Perform truncation somehow

# Draw n samples, where N > P has to hold
nSamples = maxdegree * 20
X = sampleMeasure(nSamples, op)

# Evaluate model -> vector Y
Y = model.(X)

# Build matrix Φ with ϕ(x(i))
# Φ = Array{Float64}(undef, nSamples, maxdegree+1)
Φ = [ evaluate(j, X[i], op) for i = 1:nSamples, j = 0:maxdegree]

# Ordinary least squares regression
y_reg = leastSquares(Φ, Y)
println.(y_reg)
println()


# Validation of PCE model
println("Comapre coefficients analytic <-> regression:")
println(norm(y_ana - y_reg, Inf))

# genError = empError(Y, Φ, y_reg)
# println("Determination coefficient R² (normalized empicial error): ", 1 - genError)

ϵLoo = looError(Y, Φ, y_reg)
println("Determination coefficient Q² (leave-one-out error): ", 1- ϵLoo)
println()



# ------- Monte-Carlo -------
# Evaluate model function on same set of previously drawn samples X
y_mc = model.(X)



# ------- Comparison of moments -------
println("Comparison of moments to analytic solution")

# Analytic moments for y
mean_ana = exp(μ + σ^2/2)
std_ana = sqrt(exp(2*μ + σ^2) * (exp(σ^2) - 1))
skew_ana = sqrt(exp(σ^2) - 1) * (exp(σ^2) + 2)

# MC moments
mean_mc, std_mc, skew_mc = mean(y_mc), std(y_mc), skewness(y_mc)
error_mean_mc = abs(mean_ana - mean_mc)
error_std_mc = abs(std_ana - std_mc)
println("\t\t\t error MC, mean: \t $(error_mean_mc)")
println("\t\t\t error MC, std: \t $(error_std_mc)")

# Projection moments
mean_proj = mean(y_proj, op)
std_proj = std(y_proj, op)
error_mean_proj = abs(mean_ana - mean_proj)
error_std_proj = abs(std_ana - std_proj)
println("\t\t\t error proj, mean: \t $(error_mean_proj)")
println("\t\t\t error proj, std: \t $(error_std_proj)")

# Regression moments
mean_reg = mean(y_reg, op)
std_reg = std(y_reg, op)
error_mean_reg = abs(mean_ana - mean_reg)
error_std_reg = abs(std_ana - std_reg)
println("\t\t\t error reg, mean: \t $(error_mean_reg)")
println("\t\t\t error reg, std: \t $(error_std_reg)")