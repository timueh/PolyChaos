using LinearAlgebra #TODO: thin out dependency, move to Package file

export leastSquares

# Ordinary least-squares regression
function leastSquares(Φ::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real})
    @assert size(Φ, 1) == size(Y, 1) "Dimension mismatch in matrix Φ and observation vector Y."

    # FUTURE: need to account for correlated data, if present
    
    coefficients = zeros(size(Y))

    rcond = 1/cond(Φ) # reciproce of condition number #TODO: either use method for inverse directly or use normal cond
    
    if rcond > 1e-6
        ## Fastest, but least accurate (squares condition number)
        ΦTΦ = Φ' * Φ
        coefficients = ΦTΦ \ (Φ' * Y)
    
    elseif rcond > eps()
        ## (Much) more precise but ~ twice as slow
        coefficients = Φ \ Y

    else
        ## Slowest, but best precision for very ill-conditioned matrices. For better conditioned matrices it is not as good as other methods. 
        @warn("Matrix condition number is really high. Results can become inaccurate.")
        coefficients = pinv(Φ) * Y # TODO: calibrate tolerances of pinv
    end
    
    return coefficients
end