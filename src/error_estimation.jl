import Statistics: var

# Valiadation set
    
    # Not for now.

    

# Normalized empirical error (generalization error)
# Following Blatman2009 (Dissertation)
function empError(Y::Vector{Float64}, Φ::Matrix{Float64}, pceCoeffs; adjusted = false)
    @assert length(Y) > 0           "Empty results vector Y"
    @assert length(Y) == size(Φ, 1) "Inconsistent number of elements."
    # TODO: Require column vectors

    # Compute PCE model response
    Y_Pce = Φ * pceCoeffs
    
    # Variance of sampling evaluation (true model)
    n = length(Y)
    meanY = mean(Y)
    varY = n > 1 ? 1/(n-1) * sum( (Y[i] - meanY)^2 for i in 1:n ) : 0

    # Empirical error
    empError = 1/n * sum( (Y[i] - Y_Pce[i])^2 for i in 1:n )
    
    # Normalize with variance
    nempError = varY == 0 ? 0.0 : empError / varY


    if adjusted
        # TODO: adjusted empirical error (penalize larger PCE bases)
        # p = ... #bincoeff
        # empError2 = (n-1) / (n - p - 1) * ( empError)
    end

    return nempError
end




# Leave-one-out cross-validation error
function looError(Y::Vector{Float64}, Φ::Matrix{Float64}, pceCoeffs)
    @assert length(Y) > 0           "Empty results vector Y"
    @assert length(Y) == size(Φ, 1) "Inconsistent number of elements."

    # Compute PCE model response
    Y_Pce = Φ * pceCoeffs

    # h-factor for validation sets
    M = Φ * inv(Φ' * Φ) * Φ' #TODO: fix conditioning
    N = size(M, 1)
    h = ones(N) - diag(M)
    
    looError = 1/N * sum( ( (Y[i] - Y_Pce[i]) / h[i] )^2 for i in 1:N )
    
    # Variance of sampling evaluation (true model)
    n = length(Y)
    meanY = mean(Y)
    varY = n > 1 ? 1/(n-1) * sum( (Y[i] - meanY)^2 for i in 1:n ) : 0

    println("M: ", M)
    println("h: ", h)
    println("varY: ", varY)
    println("looError: ", looError)

    # Return error normalize with variance
    varY == 0 ? 0.0 : looError / varY
end
