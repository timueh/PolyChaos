# what method is used? -> hardcoded regression for now
# if error increases 2 times in a row overfitting may occur and we have to exit or restart



function adaptiveBase(maxdeDree::Int = 10; method::String, m::Measure, model; Nrec::Int=maxDegree+1, ϵ = 1e-10)
    deg = 1
    error = 1000
    

    # TODO: check for overfitting
    # Compare to preset threshold. Stop and return lowest error PCE, if condition is met.
    while error > ϵ && deg < maxDegree
        
        # 1. Compute current orthogonal basis
        # TODO: based on provided measure
        op = GaussOrthoPoly(deg; Nrec)

        # 2. Compute problem solution for current base
        if method == "OLS"
            # sample measure
            # evaluate model, evaluate ortho polys
            # regression
            PCE = leastSquares()

        elseif method == "projection"
            # evaluate model, evaluate ortho polys
            # integrate
            PCE = ...
        else
            throw(error("method $method not implemented yet"))
        end

        # 3. Compute generalization error for convergence
        score = looError(Y, Φ, PCE)

    end
end




function test()
    op = GaussOrthoPoly()

end