using LinearAlgebra #TODO: thin out dependency, move to Package file

export leastSquares


# if Φ is a row matrix and is passed as a 1 dimensional array
# function leastSquares(Φ::AbstractVector{<:Real}, Y::AbstractVector{<:Real})
#     Φ = reshape(Φ, length(Φ), 1)
#     leastSquares(Φ, Y)
# end

# Ordinary least-squares regression
function leastSquares(Φ::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real})
    @assert size(Φ, 1) == length(Y) "Dimension mismatch in matrix Φ and observation vector Y."

    # FUTURE: need to account for correlated data, if present
    
    coefficients = zeros(size(Y))

    condition = cond(Φ) # Matrix condition number
    
    # Fastest, but least accurate (squares condition number)
    if condition < 1e6
        ΦTΦ = Φ' * Φ
        coefficients = ΦTΦ \ (Φ' * Y)
    
    # (Much) more precise but ~ twice as slow
    elseif condition < eps()
        coefficients = Φ \ Y

    # Slowest, but best precision for very ill-conditioned matrices. For better conditioned matrices it is not as good as other methods. 
    else
        @warn("Matrix condition number is really high. Results can become inaccurate.")
        coefficients = pinv(Φ) * Y # TODO: calibrate tolerances of pinv
    end
    
    return coefficients
end