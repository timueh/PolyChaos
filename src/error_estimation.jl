export empError, 
       looError

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
    nempError = varY == 0 ? 0 : empError / varY


    if adjusted
        # TODO: adjusted empirical error (penalize larger PCE bases)
        # p = ... #bincoeff
        # empError2 = (n-1) / (n - p - 1) * ( empError)
    end

    return nempError
end



# Leave-one-out cross-validation error
# Following Blatman2009 (Dissertation)
function looError(Y::Vector{Float64}, Φ::Matrix{Float64}, pceCoeffs)
    @assert length(Y) > 0           "Empty results vector Y"
    @assert length(Y) == size(Φ, 1) "Inconsistent number of elements."

    # Compute PCE model response
    Y_Pce = Φ * pceCoeffs

    # h-factor for validation sets
    # println(Φ * Φ')
    # println(inv(Φ' * Φ))
    # println(Φ * inv(Φ' * Φ))
    M = Φ * inv(Φ' * Φ) * Φ' #TODO: fix conditioning
    N = size(M, 1)
    h = ones(N) - diag(M)
    
    # Variance of sampling evaluation (true model)
    n = length(Y)
    meanY = mean(Y)
    varY = n > 1 ? 1/(n-1) * sum( (Y[i] - meanY)^2 for i in 1:n ) : 0
    
    # compute squared error with h-factor
    loo = 1/N * sum( ( (Y[i] - Y_Pce[i]) / h[i] )^2 for i in 1:N )
    
    # println("M: ", M)
    # println("Y - Y_Pce: ", Y - Y_Pce)
    # println("h: ", h)
    # println("varY: ", varY)
    # println("looError: ", loo)

    # Normalize error with variance. Set error to 0, if var is 0
    looError = varY == 0 ? 0 : loo / varY

    # Return loo error. Return Inf, if it is NaN. Can happen for underdetermined problems.
    return isnan(looError) ? Inf : looError
end
