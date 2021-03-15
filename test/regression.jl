using Test

include("../src/regression.jl") # need to change this to the package eventually


# Dimension mismatch
# M = rand(4,5)
# y = rand(6)
# @test_throws DimensionMismatch leastSquares(M,y)


## Well-conditioned matrices
tol = 1e-13
dims = [[5,20],[10,200],[20,2000]]

@testset "Regression on well-conditioned random matrices" begin
    for dim in dims
        M = rand(dim[1], dim[2])
        # x = rand(dim[2])
        # y = M*x
        y = rand(dim[1])
        x_reg = leastSquares(M, y)

        # @test isapprox(norm(x - x_reg, Inf), 0; atol = tol)
        @test isapprox(norm(M*x_reg - y, Inf), 0; atol = tol)
    end
end


## Ill-conditioned matrices
tol = 0.5
dims = [12, 16, 20]
# Hilbert matrix with bad condition number
hilbert(n) = [1 / (i + j - 1) for i in 1:n, j in 1:n]

@testset "Regression on ill-conditioned matrices" begin
    for dim in dims
        M = hilbert(dim)
        y = rand(dim)
        x_reg = leastSquares(M, y)
        @test isapprox(norm(M*x_reg - y, Inf), 0; atol = tol)
    end
end

    # dimensions okay