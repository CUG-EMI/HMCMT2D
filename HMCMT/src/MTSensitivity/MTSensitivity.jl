#-------------------------------------------------------------------------------
#
# module `MTSensitivity` defines routines to compute the sensitivity matrix.
#
# (c) Han Bo and Peng Ronghua,  Sep., 2021
#
#-------------------------------------------------------------------------------
module MTSensitivity

using Printf, SparseArrays, Statistics
using LinearAlgebra
using HMCMT.MTFwdSolver
using HMCMT.HMCFileIO
using HMCMT.HMCUtility
using MUMPS

export PreRxSens

#-------------------------------------------------------------------------------
mutable struct PreRxSens{T<:Float64}

    # the index of sites' z-loc in zNode
    zid :: Int

    # the derivative of grid nodal EM fields at (zid)th and (zid+1)th layer
    # with respect to all grid nodal EM fields (solution fields).
    dFn0 :: SparseMatrixCSC{T, Int}
    dFn1 :: SparseMatrixCSC{T, Int}

    # cell conductivity of the layer immediately below sites, and their
    # derivatives with respect to the total cell conductivity.
    sigma1 :: Vector{T}
    dsigma1 :: SparseMatrixCSC{T, Int}

    # y- cell size
    yLen :: Vector{T}

    # z- cell size of the layer immediately below sites.
    zLen1 :: T

    # linear mapping from grid nodal fields at (zid)th layer to fields at sites.
    linRxMap :: SparseMatrixCSC{T, Int}

    # linear mapping from grid midpoint fields at (zid)th layer to fields at sites.
    linRxMap2 :: SparseMatrixCSC{T, Int}

end # type


#-------------------------------------------------------------------------------
include("MT1DSensitivity.jl")
include("sensUtils.jl")
include("dataFuncSens.jl")
include("compJacMat.jl")
include("compJacTMat.jl")
include("compJacTMatVec.jl")

end
