#-------------------------------------------------------------------------------
#
# module MT2DFileIO defines routines for data and model input and output of
# MT2D problem.
#
# (c) HB, 25 Dec., 2015
# (c) Peng Ronghua, Aug., 2020 for v0.7
#
#-------------------------------------------------------------------------------
module HMCFileIO

using  Printf, SparseArrays
using  LinearAlgebra

export MTData
export TensorMesh2D
export readEMModel2D
export writeEMModel2D
export readMT2DData
export writeMT2DData

#-------------------------------------------------------------------------------
"""
    type `MTData` encapsalates data structure for 2D MT modeling.
"""
mutable struct MTData{T<:Real}

    rxLoc     :: Array{T}                   # receiver location array
    freqs     :: Vector{T}                  # frequency array
    dataType  :: String                     # data type
    dataComp  :: Vector{String}             # data component
    rxID      :: Vector{Int}                # receiver index
    freqID    :: Vector{Int}                # frequency index
    dtID      :: Vector{Int}                # datacomp index
    dataID    ::Vector{Bool}                # data index

    # what modes
    compTE :: Bool
    compTM :: Bool

end # type


#-------------------------------------------------------------------------------
mutable struct TensorMesh2D{T<:Real}

    yLen::Vector{T}             # cell width in y direction
    zLen::Vector{T}             # cell width in z direction excluding the airLayer
    airLayer::Vector{T}         # airlayer
    gridSize::Vector{Int}       # cell numbers in y,z direction
    origin::Vector{T}           # origin of mesh
    sigma::Vector{T}            # conductivity of mesh

    Face::SparseMatrixCSC{Float64, Int}       # Face area
    Grad::SparseMatrixCSC{Float64, Int}       # nodal gradient
    AveCN::SparseMatrixCSC{Float64, Int}      # averaging mapping from cell-center to node
    AveCF::SparseMatrixCSC{Float64, Int}      # averaging mapping from cell-center to face
    setup::Bool                 # whether operators have been set up or not

end # type


#-------------------------------------------------------------------------------
include("readEMModel2D.jl")
include("writeEMModel2D.jl")
include("readMT2DData.jl")
include("writeMT2DData.jl")


end # MT2DFileIO
