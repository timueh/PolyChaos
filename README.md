[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://timueh.github.io/PolyChaos.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://timueh.github.io/PolyChaos.jl/dev)
[![Build Status](https://travis-ci.org/timueh/PolyChaos.jl.svg?branch=master)](https://travis-ci.org/timueh/PolyChaos.jl)
[![codecov](https://codecov.io/gh/timueh/PolyChaos.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/timueh/PolyChaos.jl)
[![Coverage Status](https://coveralls.io/repos/github/timueh/PolyChaos.jl/badge.svg?branch=master)](https://coveralls.io/github/timueh/PolyChaos.jl?branch=master)



# PolyChaos -- Orthogonal Polynomials, Quadrature, and Polynomial Chaos

A Julia package to construct orthogonal polynomials, their quadrature rules, and use it with polynomial chaos expansions.

The package requires `Julia 1.0` or newer.
In `Julia` switch to the package manager
```julia
julia> ]
(v1.0) pkg> add PolyChaos
```
This will install PolyChaos and its dependencies.
Once that is done, load the package:
```julia
julia> using PolyChaos
```
That's it.

Let's take a look at a simple example.
We would like to solve the integral
```math
\int_0^1 6 x^5 \mathrm{d}x.
```
Exploiting the underlying uniform measure, the integration can be done exactly with a 3-point quadrature rule.
```@example mysetup
opq = OrthoPolyQ("uniform01",3)
integrate(x->6x^5,opq)
```


For more information please visit the [documentation](https://timueh.github.io/PolyChaos.jl/stable/).