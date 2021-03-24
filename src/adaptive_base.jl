# TODO: if error increases 2 times in a row overfitting may occur and we have to exit or restart
# TODO: multiple dispatch for bases on arbitrary weight functions
# TODO: check for overfitting


# TODO: what method is used? -> hardcoded regression for now

# Univariate adaptive basis algo
# Arguments:
#   * maxDegree: maximum degree searched for
#   * op: polnomial basis of PCE model
#   * model: function handle for model to evaluate
#   * method: name of method to compute PCe coefficients
# function adaptiveBasis(method::String, op::AbstractOrthoPoly, model; maxDegree::Int = 10, Nrec::Int = maxDegree+1, ϵ = 1e-10)
function adaptiveBasis(PCEmodel, op; maxDegree::Int = 10, Nrec::Int = maxDegree+1, ϵ = 1e-10)
    error = 1000 # error of current PC expansion
    bestDegree = 1 # track best degree
    deg = 1
    
    
    # Compare to preset threshold. Stop and return lowest error PCE, if condition is met.
    while error > ϵ && deg < maxDegree
        
        # 1. Compute current orthogonal basis
        currentBasis = op(deg)

        # 2. Compute problem solution for current base
        # if method == "OLS"
            # sample measure
            # evaluate model, evaluate ortho polys
            # regression
            # PCE = leastSquares()

        # elseif method == "projection"
            # evaluate model, evaluate ortho polys
            # integrate
            # PCE = 1
        # else
            # throw(error("method $method not implemented yet"))
        # end
        PCE  = PCEmodel(deg)


        # 3. Compute generalization error for convergence
        error = looError(Y, Φ, PCE)
        
        # error = ComputeError

        deg = deg + 1
        
    end
    
    #TODO: save result as sparse array?
    return PCE
end


function projectionModel(model::Function, op::AbstractOrthoPoly, maxDeg::Int)
    t2 = Tensor(2, op)

    coefficients = Array{Float64}(undef, maxDeg+1)

    for k = 0:maxDeg
        # Function for evaluation of model and basis polynomial at given quadrature nodes t
        g(t) = model(t) .* evaluate(k, t, op)
        # Perfom numerical integration for the inner product <x, ϕ_k>
        proj = integrate(g, op.quad)
        # Inner product <ϕ_k,ϕ_k> for Galerkin projection
        norm = t2.get([k,k])
        # Coefficient is the fraction of the two inner products
        coefficients[k+1] = proj / norm
    end

    return coefficients
end


# function test()
    μ = 0
    σ = 1
    model(x) = exp.(μ + σ * x)
    
    # deg = 10
    op(deg) = GaussOrthoPoly(deg, Nrec = deg * 2)

    # projection(deg) = projectionModel(model, op(deg), deg)

    # op_opti = adaptiveBasis(projection, op)
# end
