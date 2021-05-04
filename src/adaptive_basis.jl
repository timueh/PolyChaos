# TODO: Overfitting: strategy for overfitting
# TODO: multiple dispatch for bases on arbitrary weight functions

using LinearAlgebra: diag, cond
include("error_estimation.jl")


# Univariate adaptive basis algo
# Arguments:
#   * op: Function hanlde for polnomial basis
#   * modelFun: function handle for model to evaluate
#   * pceFun: function handle to compute PCE coefficients
#   * pceModel: struct storing all relevant pce information
#   * maxDegree: maximum allowed degree of basis
#   * ϵ: target error
function adaptiveBasis(opFun::Function, pceFun::Function, pceModel; maxDegree::Int = 20, ϵ = 1e-10)
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
        modelFun = pceModel.modelFun
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
mutable struct OLSModel{T<:AbstractOrthoPoly}
    op::T   # Chosen orthogonal basis FUTURE: Sparse Ortho Poly
    modelFun::Function  # Function handle for system model
    pceCoeffs::Array{Float64,1}
    error::Float64      # Error of PCE model
    # Y::Vector{Float64}  # Model evaluaions
    # Φ::Matrix{Float64}  # Moment matrix
    function OLSModel(op::AbstractOrthoPoly, modelFun::Function)
        new{typeof(op)}(op, modelFun, [], Inf)
    end
end


# Compute the PCE coefficients using OLS regression
# TODO: sample function
# TODO: determine good number of samples (-> maybe move sampling outside)
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
adaptiveBasis(op, pceFun, olsModel)



function computeED(nSamples, op::AbstractOrthoPoly)
    X = sampleMeasure(nSamples, op)
end


# Compute a sparse basis for the provided pce model and parameters
# Parameters:
#   * op: The full candidate orthogonal basis
function sparsePCE(op::AbstractOrthoPoly, modelFun::Function; Q²tgt = .8, pMax = 10, jMax = 6)
    # Parameter specification
    pMax = min(op.deg, pMax)
    COND = 1e4 # Maximum allowed matrix condition number (see Blatman2010)
    α = 0.001 # (see Blatman2010)
    ϵ = α * (1 - Q²tgt) # Error threshold of coefficients

    # 1. Build initial ED, calc Y
    sampleSize = pMax * 2 # TODO: How to detemrine?
    X = computeED(sampleSize, op)
    Y = modelFun.(X) # This is the most expensive part
    
    restart = true
    # Outer loop: Iterate on experimental design
    while restart
        restart = false
        
        # 2. Initialize
        p = 0
        Ap = [0] # Zero element
        Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in Ap ]
        pce = leastSquares(Φ, Y)
        R²0 = 1 - empError(Y, Φ, pce)
        println("R²0: $R²0")
        Q²0 = 1 - looError(Y, Φ, pce)

        # Main loop: Iterate max degree p
        while Q²0 ≤ Q²tgt && p ≤ pMax
            p += 1
            j = 0
            Q² = 0
            
            # Iterate max interactions j
            while Q² ≤ Q²tgt && j < jMax
                j += 1
                J = [] # Temporary store potential new basis elements
                candidates = [p] # FUTURE: this needs to capture all current multi indices. ALso need to change from indices to objects (?)

                # Forward step: compute R² for all candidate terms and keep the relevant ones
                for a in candidates
                    A = Ap ∪ p
                    Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in A]
                    pce = leastSquares(Φ, Y)
                    R² = 1 - empError(Y, Φ, pce) # Only need R² error here "due to more efficiency" (Blatman 2010)
                    println("R² (p=$p, j=$j): ", R²)
                    ΔR² = R² - R²0
                    # println("ΔR²: ", ΔR²)
                    if ΔR² ≥ ϵ
                        J = J ∪ a
                    end
                end
                println("Candidates J: $J")

                # FUTURE: Sort Jp and ΦJ according to ΔR²
                # J = sort(J)
                # R = []

                # Conditioning Check: If resulting enriched basis does not yield a well-conditioned moments matrix, we have to restart
                # for a in J
                A = Ap ∪ J
                Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in A]
                # check Φ
                if cond(Φ) > COND 
                    # Increase experimental design and restart computations
                    restart = true
                    sampleSize *= 2 # TODO: build properly
                    Y = computeED(sampleSize, op) #TODO: Reuse old ED data, this part is very expensive!
                    # break
                else
                    Ap_new = Ap ∪ R
                    # R = R ∪ a
                end
                # end

                Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in Ap_new]
                pce = leastSquares(Φ, Y)
                R²0 = 1 - empError(Y, Φ, pce)
                Q²0 = 1 - looError(Y, Φ, pce)
                
                
                # Backward step: Remove new polynomials one by one and compute the effect on Q²
                if !restart 
                    Del = []
                    for a in Ap_new
                        A = filter!(e->e≠a,Ap_new)
                        # New candiadte basis -> compute new determination coefficients
                        Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in A]
                        pce = leastSquares(Φ, Y)
                        R² = 1 - empError(Y, Φ, pce)
                        Q² = 1 - looError(Y, Φ, pce)

                        ΔR² = R²0 - R²

                        # If decrease in accuracy is too small, throw polynomial away
                        if ΔR² ≤ ϵ
                            Del ∪ a
                        end
                    end

                    # Update basis and compute errors for next iteration
                    Ap = filter!(e->e∉Del, Ap_new)
                    Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in A]
                    pce = leastSquares(Φ, Y)
                    R²0 = 1 - empError(Y, Φ, pce)
                    Q²0 = 1 - looError(Y, Φ, pce)
                end

            end
    
        end
    
    end
    
    return #TODO
end


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
