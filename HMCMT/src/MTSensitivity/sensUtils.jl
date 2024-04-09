export preSetRxFieldSens, linearInterpMat
# export setActiveElement

#-------------------------------------------------------------------------------
"""
    `preSetRxFieldSens(rxLoc, yNode, zNode, sigma)`

Pre-sets stuff needed to compute the partial derivative of EM fields at sites
with respect to solution fields and model conductivities. It is independent of
frequency.
-Input:
     `sigma` :: Vector, conductiviy array for 2-D grid
-Output:
     rxSensInfo :: PreRxSens

"""
function preSetRxFieldSens(rxLoc::Array{T,2}, yNode::Vector{T}, zNode::Vector{T},
    sigma::Vector{T}) where {T<:Float64}

    ny = length(yNode) - 1
    nz = length(zNode) - 1
    nNode = (ny+1)*(nz+1)
    nCell = ny*nz
    zLen = diff(zNode)

    # find out on the top of which layer the receivers are located.
    # assume all reveivers have the same z-loc (no topography).
    zRx = rxLoc[1,2]
    zid = findfirst(x -> x<0.1, abs.(zNode .- zRx))

    Inode = spunit(nNode)
    id0 = (zid-1)*(ny+1)+1:zid*(ny+1)
    id1 = zid*(ny+1)+1:(zid+1)*(ny+1)
    dFn0 = Inode[id0, :]
    dFn1 = Inode[id1, :]

    Icell = spunit(nCell)
    sigma1 = sigma[(zid-1)*ny+1:zid*ny]
    id0 = (zid-1)*ny+1:zid*ny
    dsigma1 = Icell[id0, :]

    yLen  = diff(yNode)
    zLen1 = zLen[zid]

    linRxMap = linearInterpMat(rxLoc[:,1], yNode)

    yCen = ( yNode[1:end-1] + yNode[2:end] ) / 2.0
    linRxMap2 = linearInterpMat(rxLoc[:, 1], yCen)

    return PreRxSens(zid, dFn0, dFn1, sigma1, dsigma1, yLen, zLen1, linRxMap, linRxMap2)

end


#-------------------------------------------------------------------------------
"""
    `linearInterpMat(point, x)`

gets 1D linear interpolation points and weights and use them to construct the
interpolation matrix.

"""
function linearInterpMat(point::Vector{T}, x::Vector{T}, getMat=true) where {T<:Float64}

    npts  = length(point)
    nNode = length(x)

    inds    = Array{Any}(undef, npts)
    weights = zeros(Float64, npts, 2)
    interpMat = spzeros(Float64, nNode, npts)

    for i = 1:npts
        indL, indR, wL, wR = linearInterp(point[i], x)
        inds[i] = (indL,indR)
        weights[i,:] = [wL wR]
        interpMat[:,i] = sparsevec([indL;indR], [wL;wR], nNode)
    end

    !getMat && ( return inds, weights )

    return interpMat

end

#-------------------------------------------------------------------------------
"""
    setActiveElement(nGrid, sigma, sigFix)

setup fixed elements during computation.

"""
function setActiveElement(sigma::Vector, sigFix)

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
        inaCell = sparse(1I, nGrid, nGrid)
        inaCell = inaCell[:, indFix]

        inaInd  += indFix
        nFixCell = ones(size(inaCell, 2))

        bgModel += sigFix[i] * inaCell * nFixCell

    end

    aInd = findall(inaInd .== 0)
    activeCell = activeCell[:, aInd]

    return activeCell, bgModel

end #


"""
    linearInterp(point, x)

Compute interpolation weights using linear interpolation scheme

"""
function linearInterp(point::T, x::Vector{T}) where {T}

    val, ind = findmin(abs.(point .- x))
    # location of point
    if point - x[ind] > 0
    # point on the right
        indL = ind
        indR = ind + 1
    else
    # point on the left
        indL = ind - 1
        indR = ind
    end
    # ensure interpolation points within the bound
    n = length(x)
    indL = maximum([minimum([indL, n]), 1])
    indR = maximum([minimum([indR, n]), 1])

    if indL == indR
        return indL, indR, 0.5, 0.5
    end
    # interpolation weights
    xLen = x[indR] - x[indL]
    wL   = 1 - (point - x[indL]) / xLen
    wR   = 1 - (x[indR] - point) / xLen

    return indL, indR, wL, wR

end
