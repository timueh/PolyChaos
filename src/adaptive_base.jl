# TODO: if error increases 2 times in a row overfitting may occur and we have to exit or restart
# TODO: multiple dispatch for bases on arbitrary weight functions
# TODO: check for overfitting

using LinearAlgebra: diag
include("error_estimation.jl")


# Univariate adaptive basis algo
# Arguments:
#   * maxDegree: maximum degree searched for
#   * op: polnomial basis of PCE model
#   * model: function handle for model to evaluate
#   * method: name of method to compute PCe coefficients
# function adaptiveBasis(method::String, op::AbstractOrthoPoly, model; maxDegree::Int = 10, Nrec::Int = maxDegree+1, ϵ = 1e-10)
function adaptiveBasis(PCEmodel, op; maxDegree::Int = 20, Nrec::Int = maxDegree+1, ϵ = 1e-10)
    error = Inf # error of current PC expansion
    deg = 1     # current degree
    deg_opti = 1 # track best degree
    # op_opti =      # best TODO: return basis
    PCE = []    # PCE coefficients for best basis

    
    # Compare to preset threshold. Stop and return lowest error PCE, if condition is met.
    while error > ϵ && deg < maxDegree
        
        # 1. Compute current orthogonal basis
        op_current = op(deg)

        # 2. Compute PCE coefficients for current base
        currentModel = PCEmodel(op_current)
        PCE  = currentModel.coefficients

        # 3. Compute generalization error for convergence
        error = currentModel.error

        # 4. Check if result increases, store best
        deg_opti = deg
        deg = deg + 1
    end
    
    #TODO: save result as sparse array?
    return PCE, error, deg_opti
end


# maybe shift this part to regression file??
struct OLSModel{O, S, C}
    op::AbstractOrthoPoly   # Chosen orthogonal basis
    sysFun::Function        # System equations of the model as function handle
    coefficients::Array{Float64,1}
    error::Float64
    # TODO: sample function
    # TODO: (number of samples)
    
    # inner constructor    
    function OLSModel(op::AbstractOrthoPoly, sysFun::Function, errorFun::Function; compError = true)
        deg = op.deg
        # Draw n samples from the underlying measure
        nSample = deg * 20
        X = sampleMeasure(nSample, op)
        # Build moment matrix Φ for nSample samples and deg basis polynomials
        Φ = [ evaluate(j, X[i], op) for i = 1:nSample, j = 0:deg]
        # Evaluate the model
        Y = sysFun.(X)
        # Perform OLS regression
        coefficients = leastSquares(Φ, Y)
        
        # compute the leave-one-out error on the result
        compError ? error = errorFun(Y, Φ, coefficients) : error = nothing
        
        new{typeof(op), typeof(sysFun), typeof(coefficients)}(op, sysFun, coefficients, error)
    end
end



# Define the model
μ = 0
σ = 1
model(x) = exp.(μ + σ * x)

# Type of underlying basis
# TODO what is iwth general ortho poly? -- multiple dispatch
# deg = 10
op(deg) = GaussOrthoPoly(deg, Nrec = deg * 2)

# Compute adaptive basis with a projection approach
# projection(deg) = projectionModel(model, op(deg), deg)
    # op_opti = adaptiveBasis(projection, op)
    
errorFun(Y, Φ, PCE) = looError(Y, Φ, PCE)
    
# Compute adaptive basis with a regression approach
ols(basis) = OLSModel(basis, model, errorFun)
op_opti = adaptiveBasis(ols, op)



# function projectionModel(model::Function, op::AbstractOrthoPoly, maxDeg::Int)
#     t2 = Tensor(2, op)

#     coefficients = Array{Float64}(undef, maxDeg+1)

#     for k = 0:maxDeg
#         # Function for evaluation of model and basis polynomial at given quadrature nodes t
#         g(t) = model(t) .* evaluate(k, t, op)
#         # Perfom numerical integration for the inner product <x, ϕ_k>
#         proj = integrate(g, op.quad)
#         # Inner product <ϕ_k,ϕ_k> for Galerkin projection
#         norm = t2.get([k,k])
#         # Coefficient is the fraction of the two inner products
#         coefficients[k+1] = proj / norm
#     end

#     return coefficients
# end