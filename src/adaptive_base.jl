# TODO: Overfitting: strategy for overfitting
# TODO: multiple dispatch for bases on arbitrary weight functions
# TODO: no use of abstract types in ols model

using LinearAlgebra: diag
include("error_estimation.jl")

# Univariate adaptive basis algo
# Arguments:
#   * op: Function hanlde for polnomial basis
#   * modelFun: function handle for model to evaluate
#   * pceFun: function handle to compute PCE coefficients
#   * pceModel: struct storing all relevant pce information
#   * maxDegree: maximum allowed degree of basis
#   * ϵ: target error
function adaptiveBasis(opFun::Function, modelFun::Function, pceFun::Function, pceModel; maxDegree::Int = 20, ϵ = 1e-10)
    error = Inf     # error of current PC expansion
    error_opt = Inf    # error for optimal solution
    deg = 1         # current degree
    badFit = false  # overfitting flag
    
    # Compare to preset threshold. Stop and return lowest error PCE, if condition is met.
    while error > ϵ && deg < maxDegree
        # 1. Compute current orthogonal basis
        # FUTURE: truncation scheme
        op_current = opFun(deg)
        
        # 2. Compute new PCE coefficients and error for current basis
        # computePCE(PCEmodel, true)
        pce, error = pceFun(op_current, modelFun)

        # 3. Check if result increases, store best, check for overfitting
        println("Current error: $error \t minimal error: $error_opt")
        if error < error_opt
            # Update optimal degree, error and pce coefficients
            error_opt = error
            pceModel.op = op_current
            pceModel.pceCoeffs = pce
            pceModel.error = error
            badFit = false
        elseif badFit
            # No increase in 2 iterations -> stop
            println("break")
            break
        else
            badFit = true
        end

        deg = deg + 1
        println(deg)
    end   
end


# Struct, containing all relevant data for OLS regression for PCE estimation
# TODO: no abstract types
mutable struct OLSModel
    op::AbstractOrthoPoly   # Chosen orthogonal basis FUTURE: Sparse Ortho Poly
    modelFun::Function  # Function handle for system model
    pceCoeffs::Array{Float64,1}
    error::Float64      # Error of PCE model
    # Y::Vector{Float64}  # Model evaluaions
    # Φ::Matrix{Float64}  # Moment matrix
    function OLSModel(op::AbstractOrthoPoly, modelFun::Function)
        new(op, modelFun, [], Inf)
    end
end


# Compute the PCE coefficients using OLS regression
# TODO: sample function
# TODO: (number of samples)
# function computePCE(olsModel::OLSModel, compError)
function computePceOls(op::AbstractOrthoPoly, modelFun::Function; compError=true)
    deg = op.deg
    # Draw n samples from the underlying measure
    nSample = deg * 20
    X = sampleMeasure(nSample, op)
    # Build moment matrix Φ for nSaopmple samples and deg basis polynomials
    Φ = [ evaluate(j, X[i], op) for i = 1:nSample, j = 0:deg]
    # Evaluate the model
    Y = modelFun.(X)
    # Perform OLS regression
    coefficients = leastSquares(Φ, Y)

    if compError
        error = looError(Y, Φ, coefficients)
    else
        error = nothing
    end

    return coefficients, error
end


# Compute the PCE coefficients using a Galerkin projection
function computePceProjection(op::AbstractOrthoPoly, modelFun::Function; compError=true)
end


#################################
# Running the code 
#################################

# Define the model
μ = 0
σ = 1
model(x) = exp.(μ + σ * x)

# Function handle for basis generation
op(deg) = GaussOrthoPoly(deg, Nrec = deg * 2)

# Function handle for computing pce components with ols regression
pceFun(op, model) = computePceOls(op, model)

# Initialize OLSModel struct
op1 = op(1)
olsModel = OLSModel(op1, model)

# Compute adaptive basis with a regression approach
adaptiveBasis(op, model, pceFun, olsModel)



#################################
# Projection model old
#################################

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


#################################
# Regression model old
#################################

# # maybe shift this part to regression file??
# struct OLSModel{O, S, C}
#     op::AbstractOrthoPoly   # Chosen orthogonal basis
#     sysFun::Function        # System equations of the model as function handle
#     coefficients::Array{Float64,1}
#     error::Float64
#     # TODO: sample function
#     # TODO: (number of samples)
    
#     # inner constructor    
#     function OLSModel(op::AbstractOrthoPoly, sysFun::Function, errorFun::Function; compError = true)
#         deg = op.deg
#         # Draw n samples from the underlying measure
#         nSample = deg * 20
#         X = sampleMeasure(nSample, op)
#         # Build moment matrix Φ for nSaopmple samples and deg basis polynomials
#         Φ = [ evaluate(j, X[i], op) for i = 1:nSample, j = 0:deg]
#         # Evaluate the model
#         Y = sysFun.(X)
#         # Perform OLS regression
#         coefficients = leastSquares(Φ, Y)
        
#         # compute the leave-one-out error on the result
#         compError ? error = errorFun(Y, Φ, coefficients) : error = nothing
        
#         new{typeof(op), typeof(sysFun), typeof(coefficients)}(op, sysFun, coefficients, error)
#     end
# end


#################################
# Adaptive algo old
#################################

# Univariate adaptive basis algo
# Arguments:
#   * maxDegree: maximum degree searched for
#   * op: polnomial basis of PCE model
#   * model: function handle for model to evaluate
#   * method: name of method to compute PCe coefficients
# function adaptiveBasis(method::String, op::AbstractOrthoPoly, model; maxDegree::Int = 10, Nrec::Int = maxDegree+1, ϵ = 1e-10)
# function adaptiveBasis(PCEmodel, op; maxDegree::Int = 20, Nrec::Int = maxDegree+1, ϵ = 1e-10)
#     error = Inf     # error of current PC expansion
#     error_opt = Inf    # error for optimal solution
#     deg = 1         # current degree
#     deg_opt = 1     # degree of optimal solution
#     op_opt = 0      # basis of best solution
#     PCE_opt = []    # PCE coefficients of optimal solution
#     badFit = false  # overfitting flag

    
#     # Compare to preset threshold. Stop and return lowest error PCE, if condition is met.
#     while error > ϵ && deg < maxDegree
        
#         # 1. Compute current orthogonal basis
#         op_current = op(deg)

#         # 2. Compute new PCE model for current basis
#         currentModel = PCEmodel(op_current)
#         # Get conmputed PCE coefficients from model
#         PCE  = currentModel.coefficients

#         # 3. Compute generalization error for convergence
#         error = currentModel.error

#         # 4. Check if result increases, store best
#         println("Current error: $error \t minimal error: $error_opt")
#         if error < error_opt
#             # Update optimal degree, error and pce coefficients
#             deg_opt = deg
#             error_opt = error
#             op_opt = op_current
#             PCE_opt = PCE
#             badFit = false
#         elseif badFit
#             # No increase in 2 iterations -> stop
#             println("break")
#             break
#         else
#             badFit = true
#         end

#         deg = deg + 1
#         println(deg)
#     end
    
#     return PCE_opt, op_opt, error_opt, deg_opt
# end
