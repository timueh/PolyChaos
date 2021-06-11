using PolyChaos, Test
include("../src/sparse_pce.jl")

# Univariate test models and their corresponsing ideal bases
μ, σ = 0, 1
# model_1(x) = exp(μ + σ * x)
# model_2(x) = x^7 - 21 * x^5 + 105 * x^3 - 105 * x + x^3 - 3 * x + 1 # 0th, 3th and 7th Hermite polynomial
# model_3(x) = x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945 # He_10
# model_4(x) = x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945 + x^3 - 3* x # He_10 + He 3
model_5(x) = 0.0
model_6(x) = 1.0
model_7(x) = 50 + 5/2 * (5 * x^3) + 13/16 * (231 * x^6 - 315 * x^4 + 105 * x^2 - 5)

models = [
        #   model_1,
        #   model_2,
        #   model_3,
        #   model_4,
          model_5,
          model_6,
          model_7
         ]
fullBases =
    [
        # GaussOrthoPoly(10),
        # GaussOrthoPoly(10),
        # GaussOrthoPoly(10),
        # GaussOrthoPoly(10),
        GaussOrthoPoly(5),
        GaussOrthoPoly(5),
        LegendreOrthoPoly(10)
    ]
resultBases =  [
        #   [0,1,2,3,4,5,6],
        #   [0, 3, 7], 
        #   [10], 
        #   [3, 10], 
          [0],
          [0],
          [0, 3, 6]
         ]
# PCEs =   [[1.6483774649263525, 1.6490473953964484, 0.8219974064089732, 0.2752047708599591, 0.06663165539581808, 0.013745562538423796, 0.0017708687329313392, 0.00028892409055291307],
        #   [1,1,1],[3,1]]


# tol = 1e-7
# Parameters for sparse algorithm
pMax = 10
Q²tgt = .99999

@testset "Sparse PCE, univariate bases" begin
    for i in 1:length(models)
        model = models[i]
        op = fullBases[i]
        basis_ref = resultBases[i]
        # pce_ref = PCEs[i]
        pce, basis, maxDeg, R², Q² = sparsePCE(op, model, Q²tgt=Q²tgt, pMax=pMax, jMax=1)

        @test Q² >= Q²tgt
        @test basis == basis_ref
        # @test all(isapprox.(pce, pce_ref; atol = tol))
    end
end




# TODO: Test ill-conditioning of matrix with restarting for larger sample set 


# TODO: Test multivariate basis
@testset "Sparse PCE, multivariate bases" begin

end