#-------------------------------------------------------------------------------
# module `HMCStruct` defines data structures for HMC Bayesian inversion.
#
#-------------------------------------------------------------------------------
module HMCStruct

using  SparseArrays
using  HMCMT.HMCUtility
using  HMCMT.HMCFileIO
using  HMCMT.MTFwdSolver

export HMCPrior, initHMCPrior
export HMCParameter, initHMCParameter
export HMCStatus, initHMCStatus
export InvDataModel, setupInverseDataModel

#-------------------------------------------------------------------------------
mutable struct HMCPrior{Ti<:Int, Tv<:Float64}

    #
    burninsamples::Ti
    totalsamples::Ti

    sigBounds::Vector{Tv}     # conductivity lower/upper bounds
    sigmastd::Tv              # standard derivation associated with variables

    # prior for hmc sampler
    dt::Tv
    timestep::Vector{Ti}      # lower/upper bounds of the time step

    linearSolver::String      # linear solver type: julia builtin solver OR mumps
    massType::String          # mass matrix type: diagonal OR nondiagonal
    regParam::Tv              # regularization parameter
    nfevals::Ti

end


#-------------------------------------------------------------------------------
mutable struct HMCParameter{Ti<:Int, Tv<:Float64}

    #
    nparam::Ti                   # number of free parameters
    rhomodel::Vector{Tv}         # conductivity model
    momentum::Vector{Tv}         # momentum parameter
    #
    invM::AbstractMatrix{Tv}             # inverse of the mass matrix
    sqrtM::AbstractMatrix{Tv}            # square root of the mass matrix

end


#-------------------------------------------------------------------------------
"""
    struct `HMCStatus` records proposal statistics of HMC process.

"""
mutable struct HMCStatus{Ti<:Int, Tv<:Float64}

    #
    nAccept::Ti
    nReject::Ti
    acceptstats::Vector{Bool}   # acceptance statistics

    #
    hmstats::Array{Tv}       # [dataMisfit,mnorm,ke,he] statistics

end

#-------------------------------------------------------------------------------
"""
 struct `InvDataModel` encapusalates observed data, data weighting.

"""
mutable struct InvDataModel{T<:Float64}

    # observed data and data weighting
    obsData::Vector{ComplexF64}         # observed data
    dataW::SparseMatrixCSC              # data weighting

    # starting model
    strModel::Vector{T}                 # starting model
    refModel::Vector{T}                 # prior model

    # active cell and background model
    activeCell::SparseMatrixCSC         #
    bgModel::Vector{T}                  # background model keeping constant
    Wm::SparseMatrixCSC                 # model covariance


end


#-------------------------------------------------------------------------------
"""
    `setupInverseDataModel(mtMesh,sigma,sigFix,sigLB,sigUB,obsData,dataErr,fixIndex)`

"""
function setupInverseDataModel(mtMesh::TensorMesh2D, sigFix::Vector{T}, sigLB::T,
                               sigUB::T, obsData::Vector, dataErr::Vector{T},
                               fixIndex=zeros(0)) where {T<:Float64}

    #
    sigma = mtMesh.sigma
    activeCell, bgModel = setActiveElement(sigma, sigFix, fixIndex)
    dataW = compDataWeightMat(obsData, dataErr)
    strModel = transpose(activeCell) * sigma

    # if isinf(sigLB) || isinf(sigUB)
    #     strModel = log.(strModel)
    # else
    #     strModel = getBoundedModel(strModel, sigLB, sigUB)
    # end
    strModel = log.(strModel)
    refModel = copy(strModel)

    #
    cGrad = getCellGradient2D(mtMesh)
    cGrad = cGrad  * activeCell
    Wm    = cGrad' * cGrad
    invParam = InvDataModel(obsData,dataW,strModel,refModel,activeCell,bgModel,Wm)

    return invParam

end


#-------------------------------------------------------------------------------
function initHMCPrior()

    dt = 0.01
    timestep = [10, 15]
    linearSolver = ""
    massType = "diagonal"
    regParam = 1.0

    hmcprior = HMCPrior(100,500,[0.01,10],0.05,dt,timestep,linearSolver,massType,regParam,0)
    return hmcprior

end


#-------------------------------------------------------------------------------
function initHMCStatus(nsamples::Int)

    acceptstats = zeros(Bool, nsamples)

    #
    hmstats = zeros(4,nsamples+1)

    hmcstats = HMCStatus(0,0,acceptstats,hmstats)

    return hmcstats

end


#-------------------------------------------------------------------------------
"""
    `initHMCParameter(nparam)`

"""
function initHMCParameter(nparam::Int)

    #
    rhomodel   = zeros(nparam)
    momentum   = zeros(nparam)
    invM       = zeros(nparam, nparam)
    sqrtM      = zeros(nparam, nparam)
    hmcParam = HMCParameter(nparam,rhomodel,momentum,invM,sqrtM)

    return hmcParam

end


end # module
