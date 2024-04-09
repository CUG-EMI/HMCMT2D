#-------------------------------------------------------------------------------
#
# module `HMCUtility` defines utility routines for HMC Bayesian inversion.
#
#-------------------------------------------------------------------------------
module HMCUtility

using Random, SparseArrays, LinearAlgebra

export unirandInteger
export unirandDouble
export getGaussianProbability
export modelTransform
export compDataWeightMat
export getDataMisfit
export setActiveElement
export getBoundedModel
export avnc, spdiag

#-------------------------------------------------------------------------------
"""
    `unirandInteger(vmin, vmax)`

returns a random integer between `vmin` and `vmax`

"""
function unirandInteger(vmin::Int, vmax::Int)

    rval = rand(vmin:vmax)
    return rval

end


#-------------------------------------------------------------------------------
"""
    `randDouble(vmin, vmax)`

returns a random float number between `vmin` and `vmax`

"""
function unirandDouble(vmin::Float64, vmax::Float64)

    rval = vmin + (vmax - vmin) * rand()
    return rval

end


#-------------------------------------------------------------------------------
"""
    `getGaussianProbability(sigma, phi)`

"""
function getGaussianProbability(sigma::Float64, phi::Float64)

    #
    alpha = exp(-phi^2 / (2 * sigma^2) ) / (sqrt(2*pi) * sigma)
    return alpha
end


#-------------------------------------------------------------------------------
"""
    `modelTransform(sigModel)`

convert inverse model in logrithm domain back into linear conductivity
"""
function modelTransform(sigModel::Vector)

    sigma  = exp.(sigModel)
    dsigma = exp.(sigModel)
    dsigma = sparse(Diagonal(dsigma))

    return sigma, dsigma

end


#-------------------------------------------------------------------------------
"""
    avnc(n)

Form 1D averaging matrix from node to cell-center

"""
function avnc(n::Int)
    avn = spdiag((0.5*ones(n), 0.5*ones(n)), (0, 1), n, n+1)

    return avn
end


"""
    spdiag(d1, d2, (x1, x2), m, n)

"""
function spdiag((x1,x2), (d1,d2), m, n)
    I, J, V = SparseArrays.spdiagm_internal(d1 => x1, d2 => x2)
    return sparse(I, J, V, m, n)

end



#-------------------------------------------------------------------------------
"""
    `modelTransform(sigModel, sigLB, sigUB)`

convert inverse model in logrithmic domain back into linear conductivity with
specified upper and lower bounds (see Kim and Kim, 2011)

"""
function modelTransform(sigModel::Vector{T}, sigLB::T, sigUB::T) where {T<:Float64}

    # constant cp controls steepness of the transformed space
    cp = 2
    np = ones(length(sigModel))

    # m = [a + b * exp(px)] / [1 + exp(px)]
    m = exp.(cp * sigModel)
    p = sigUB * m
    BLAS.axpy!(sigLB, np, p)
    BLAS.axpy!(1.0, np, m)

    sigma = p ./ m

    # dm = [cp * (b - m) * (m - a)] / [b - a]
    b = np * sigUB - sigma
    a = sigma - np * sigLB
    dsigma = a .* b

    dsigma = dsigma * (cp / (sigUB - sigLB))
    spdsigma = sparse(Diagonal(dsigma))

    return sigma, spdsigma

end


#-------------------------------------------------------------------------------
"""
    `getBoundedModel(sigModel, sigLB, sigUB)`

transforms model within the model bounds.
 *sigLB = minimum conductivity value in linear scale
 *sigUB = maximum conductivity value in linear scale 

"""
function getBoundedModel(sigModel::Vector{T}, sigLB::T, sigUB::T) where {T<:Float64}

    cp = 2
    sigma = (sigModel .- sigLB) ./ (sigUB .- sigModel)
    sigma = 1/cp * log.(sigma)

    return sigma

end


#-------------------------------------------------------------------------------
"""
    `compDataWeightMat(obsData, dataErr, errFloor, errTol)`

gets data weigthing matrix.

"""
function compDataWeightMat(obsData::Vector, dataError::Vector,
                           errFloor = 0.02, errTol = 0.15)

    #
    dataError = abs.(dataError)

     # relative error floor
    #  minErr = errFloor * abs(obsData)
    #  id     = find(dataError .< minErr)
    #  dataError[id] = minErr[id]

    dataW = 1 ./ dataError

    # maximum error derivation allowed
    # relErr   = abs.(dataError ./ abs.(obsData))
    # id       = findall(relErr .> errTol)
    # dataW[id] .= 0

    dataW = sparse(Diagonal(dataW))

    return dataW

end


#-------------------------------------------------------------------------------
"""
    `getDataMisfit(dataRes)`

computes data misfit.

"""
function getDataMisfit(dataRes::Array)

    datMisfit = 0.5 * dataRes' * dataRes
    datMisfit = real(datMisfit)

    return datMisfit[1]

end


#-------------------------------------------------------------------------------
"""
    setActiveElement(nGrid, sigma, sigFix)

setup fixed elements during computation.

"""
function setActiveElement(sigma::Vector, sigFix::Vector, fixIndex=zeros(0))

    nGrid = length(sigma)

    # setup active and inactive cells for inversion
    activeCell = sparse(1I, nGrid, nGrid)

    # number of fixed conductivity
    nFix   = length(sigFix)
    inaInd = zeros(Int, nGrid)

    # background conductivity
    bgModel = zeros(nGrid)

    for i = 1:nFix

        indFix  = (sigma .== sigFix[i])
        if isempty(findall(indFix)); continue; end

        inaCell = sparse(1I, nGrid, nGrid)
        inaCell = inaCell[:, indFix]

        inaInd  += indFix
        nFixCell = ones(size(inaCell, 2))

        bgModel += sigFix[i] * inaCell * nFixCell

    end

    # indexes for fixed layer
    if !isempty(fixIndex)
        inaInd[fixIndex]  .= 1
        bgModel[fixIndex] = sigma[fixIndex]

    end

    aInd = findall(inaInd .== 0)
    activeCell = activeCell[:, aInd]

    return activeCell, bgModel

end


end # HMCUtility