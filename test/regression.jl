using PolyChaos, Test, DelimitedFiles
import LinearAlgebra: norm


# Dimension mismatch
@testset "Dimension missmatch" begin
    M = rand(4,5) 
    y = rand(6)
    @test_throws AssertionError leastSquares(M,y)
end


## Well-conditioned matrices
# dimensions = [[5,20],[10,200],[20,2000]]
tol = 1e-10
@testset "Regression on well-conditioned random matrices" begin
    for size in ["small", "medium", "large"]
        M = readdlm("test/matrices/random_$size.txt")
        x = readdlm("test/matrices/x_$size.txt")
        x = vec(x)
        y = M*x
        x_reg = leastSquares(M, y)
        @test isapprox(norm(M*x_reg - y, Inf), 0; atol = tol)
    end
end


## Ill-conditioned matrices
tol = 0.5
dims = [12, 20]
# Hilbert matrix with bad condition number
hilbert(n) = [1 / (i + j - 1) for i in 1:n, j in 1:n]

@testset "Regression on ill-conditioned matrices" begin
    for dim in dims
        M = hilbert(dim)
        y = rand(dim)
        @test_logs (:warn, "Matrix condition number is really high. Results can become inaccurate.") leastSquares(M, y)
        x_reg = leastSquares(M, y)
        @test isapprox(norm(M*x_reg - y, Inf), 0; atol = tol)
    end
end