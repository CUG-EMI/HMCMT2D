#-------------------------------------------------------------------------------
module MTFwdSolver

using HMCMT.HMCFileIO

using Printf, SparseArrays
using LinearAlgebra
using MUMPS

#
export getMT2DPredData
#-------------------------------------------------------------------------------
"""
    `getMT2DPredData(mtData, mtMesh)`

get predicted data by 2D MT modeling.

"""
function getMT2DPredData(mtData::MTData, mtMesh::TensorMesh2D)

    #

    # convert resistivity from logarithmic scale to linear scale
    rho2d = 10 .^ rho2d
    sigma = 1 ./ rho2d
    sigAir = 1e-8
    ny   = mtMesh.gridSize[1]
    nAir = length(mtMesh.airLayer)
    airMat = ones(ny,nAir) * sigAir
    sigma  = vcat(vec(airMat), sigma)
    mtMesh.sigma = copy(sigma)

    # set up 2D operator if necessary
    if !mtMesh.setup
        setupTensorMesh2D!(mtMesh)
    end

    predData = MTFwdSolver(mtMesh, mtData, linearSolver="mumps")

    return predData

end

#-------------------------------------------------------------------------------
# 2D operators
include("MT2DOperators.jl")
include("MT2DFwdSolver.jl")

# forward modeling
include("mt2DTE.jl")
include("mt2DTM.jl")
include("mt1DField.jl")


end # module
