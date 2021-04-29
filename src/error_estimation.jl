import Statistics: var

# Valiadation set
    
    # Not for now.

    

# Normalized empirical error (generalization error)
# Following Blatman2009 (Dissertation)
function empError(Y, Φ, pceCoeffs; adjusted = false)
    @assert size(Y, 1) == size(Φ, 1) "Inconsistent number of elements."
    # TODO: Require column vectors

    n = length(Y)

    # Compute PCE model response
    Y_Pce = Φ * pceCoeffs

    # Variance of sampling evaluation (true model)
    meanY = mean(Y)
    varY = 1/(n-1) * sum( (Y[i] - meanY)^2 for i in 1:n )
    println("VarY = ", varY)

    # Empirical error
    empError = 1/n * sum( (Y[i] - Y_Pce[i])^2 for i in 1:n )
    println("Empirical Error = ", empError)
    
    # Normalize with variance
    if varY == 0
        nempError = 0
    else
        nempError = empError / varY
    end


    if adjusted
        # TODO: adjusted empirical error (penalize larger PCE bases)
        # p = ... #bincoeff
        # empError2 = (n-1) / (n - p - 1) * ( empError)
    end

    return nempError
end




# Leave-one-out cross-validation error
function looError(Y, Φ, pceCoeffs)
    # consistency checks

    # Compute PCE model response
    Y_Pce = Φ * pceCoeffs

    # h-factor for validation sets
    phiMphi = Φ * inv(Φ' * Φ) * Φ' #TODO: fix conditioning
    N = size(phiMphi, 1)
    h = ones(N) - diag(phiMphi)
    
    looError = 1/N * sum( ( (Y[i] - Y_Pce[i]) / h[i] )^2 for i in 1:N )
    varY = var(Y)

    ϵLoo =  looError / varY
    
    return ϵLoo
end
