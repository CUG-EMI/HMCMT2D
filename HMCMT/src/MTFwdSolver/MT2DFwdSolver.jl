#-------------------------------------------------------------------------------
#
# module MTFwdSolver defines routines to solve MT 2D forward modeling problem.
#
# (c) Han Bo, Jan 2016
# (c) Peng Ronghua, revised at 25 Mar, 2017
#
#-------------------------------------------------------------------------------
export MT2DFwdSolver
export compMT2DTE, getBoundaryMT2DTE, compFieldsAtRxTE, compMTRespTE
export compMT2DTM, getBoundaryMT2DTM, compFieldsAtRxTM, compMTRespTM
export mt1DAnalyticField, mt1DImpedance, compMT1DEField
export CoeffMat, RxProjMat
export getBoundaryIndex, compRxProjectionTerm
export mumpsSolver
export MT2DFwdData

#-------------------------------------------------------------------------------
mutable struct CoeffMat{T<:Float64}

    # inner coefficient matrix, real and imaginary parts
    rAii :: SparseMatrixCSC{T}
    iAii :: SparseMatrixCSC{T}

    # outter (boundary) coefficients, real and imaginary parts
    rAio :: SparseMatrixCSC{T}
    iAio :: SparseMatrixCSC{T}

end # type


#-------------------------------------------------------------------------------
mutable struct RxProjMat{T<:Float64}

    #
    zGrad   :: SparseMatrixCSC{T}
    nodeMat :: SparseMatrixCSC{T}  # nodal interpolation matrix
    edgeMat :: SparseMatrixCSC{T}  # edge  interpolation matrix

end


#-------------------------------------------------------------------------------
mutable struct MT2DFwdData{T<:ComplexF64}

    exTE::Array{T}
    hxTM::Array{T}

    AinvTE::Vector
    AinvTM::Vector
    linearSolver::String

end


#-------------------------------------------------------------------------------
"""
`MTFwdSolver` is the top function being called to solve the MT2D forward problem.

Input:
    `mtMesh` =::TensorMesh2D
    `mtData`  =::MTData
    `linearSolver`: a string to determine which linear solver is used

Output:
    predData:  forward data
    exte:      solution Ex fields (at grid nodes) of TE mode
    hxtm:      solution Hy fields (at grid nodes) of TM mode
    AinvTE:    system matrix decomposition factor of TE mode
    AinvTM:    system matrix decomposition factor of TM mode
    fwdResp:   pure forward response

"""
function MT2DFwdSolver(mtMesh::TensorMesh2D, mtData::MTData; linearSolver::String="")

    MU0 = 4 * pi * 1e-7

    # extract field values from input derived type
    yLen     = mtMesh.yLen
    zLen     = mtMesh.zLen
    sigma    = mtMesh.sigma

    freqs    = mtData.freqs
    rxLoc    = mtData.rxLoc
    dataComp = mtData.dataComp
    dataType = mtData.dataType
    rxID     = mtData.rxID
    freqID   = mtData.freqID
    dtID     = mtData.dtID
    compTE   = mtData.compTE
    compTM   = mtData.compTM

    nFreq = length(freqs)
    nRx   = size(rxLoc, 1)

    # pre-set some operators
    ny = length(yLen)
    nz = length(zLen)
    nNode = (ny+1)*(nz+1)

    mu = MU0 * ones(ny*nz)
    F     = mtMesh.Face
    Grad  = mtMesh.Grad
    AveCN = mtMesh.AveCN
    AveCF = mtMesh.AveCF

    # set the index of inner and outter/boundary part of coefficient matrix
    (ii, io) = getBoundaryIndex(ny, nz)

    # resolved fields at grid nodes
    exte = zeros(ComplexF64, nNode, nFreq)
    hxtm = copy(exte)

    # MUMPS matrix decompostion factor
    if isempty(linearSolver)
        AinvTE = Array{Any}(undef,nFreq)
        AinvTM = Array{Any}(undef,nFreq)
    elseif lowercase(linearSolver) == "mumps"
        AinvTE = Array{MUMPSfactorization}(undef, nFreq)
        AinvTM = copy(AinvTE)
    end

    if compTE
        MsigCN = AveCN * (F * sigma)
        MsigCN = sparse(Diagonal(MsigCN))
        MmuF   = AveCF * (F *(1 ./ mu))
        MmuF   = sparse(Diagonal(MmuF))
        dGrad  = Grad' * MmuF * Grad

        rAii = dGrad[ii, ii]
        rAio = dGrad[ii, io]
        iAii = MsigCN[ii, ii]
        iAio = MsigCN[ii, io]

        coeMatTE = CoeffMat(rAii, iAii, rAio, iAio)

        respTE = zeros(Float64,nFreq*nRx,2)

        # loop over frequencies
        for j = 1:nFreq
            freq = freqs[j]
            js = (j-1)*nRx+1
            je = j*nRx
            (respTE[js:je,:],exte[:, j],AinvTE[j])= compMT2DTE(freq, mtMesh,
            coeMatTE, rxLoc, dataType, linearSolver=linearSolver)
        end
    end  # compTE

    if compTM
        MmuCN = AveCN * (F * mu)
        MmuCN = sparse(Diagonal(MmuCN))
        MsigF = AveCF * (F *(1 ./ sigma))
        MsigF = sparse(Diagonal(MsigF))
        dGrad = Grad' * MsigF * Grad

        rAii = dGrad[ii, ii]
        rAio = dGrad[ii, io]
        iAii = MmuCN[ii, ii]
        iAio = MmuCN[ii, io]

        coeMatTM = CoeffMat(rAii, iAii, rAio, iAio)
        respTM   = zeros(Float64,nFreq*nRx,2)

        # loop over frequencies to...
        for j = 1:nFreq
            freq = freqs[j]
            js = (j-1)*nRx+1
            je = j*nRx
            (respTM[js:je,:],hxtm[:, j],AinvTM[j]) = compMT2DTM(freq, mtMesh,
            coeMatTM, rxLoc, dataType, linearSolver=linearSolver)
        end
    end  # compTM

    # predicted data
    if occursin("Impedance", dataType)

        if compTE & !compTM
            predData = complex.(respTE[:, 1], respTE[:, 2])

        elseif !compTE & compTM
            predData = complex.(respTM[:, 1], respTM[:, 2])

        elseif compTE & compTM
            predDataTE = complex.(respTE[:, 1], respTE[:, 2])
            predDataTM = complex.(respTM[:, 1], respTM[:, 2])
            predData = hcat(predDataTE, predDataTM)
            predData = vec(transpose(predData))

        end

    elseif occursin("Rho_Pha", dataType)

        if compTE & !compTM
            predData = vec(transpose(respTE))

        elseif !compTE & compTM
            predData = vec(transpose(respTM))

        elseif compTE & compTM
            predDataTE = transpose(respTE)
            predDataTM = transpose(respTM)
            predData = vcat(predDataTE, predDataTM)
            predData = vec(predData)

        end

    end

    dataID   = mtData.dataID
    predData = predData[dataID]

    # forward response
    fwdInfo = MT2DFwdData(exte, hxtm, AinvTE, AinvTM, linearSolver)
    return predData, fwdInfo

end


#-------------------------------------------------------------------------------
"""
    `getBoundaryIndex(ny, nz)`

get the index of inner and outter/boundary part of coefficient matrix, used to
split the unknown electric fields into a boundary part and an inner part.

"""
function getBoundaryIndex(ny::Int, nz::Int)

    nNode = (ny+1) * (nz+1)

    # set the index of inner and outter/boundary part of coefficient matrix
    idx2D = (reshape(collect(1:nNode), ny+1, nz+1))'
    # inner field index
    ii = reshape((idx2D[2:end-1, 2:end-1])', (ny-1)*(nz-1))
    # top boundary field index
    it = reshape(idx2D[1,:], ny+1)
    # left boundary field index
    il = idx2D[2:end,1]
    # right boundary field index
    ir = idx2D[2:end,end]
    # bottom boundary field index
    ib = reshape(idx2D[end,2:end-1], ny-1)
    # boundary field index
    io = [it; il; ir; ib]

    return ii, io

end

#-------------------------------------------------------------------------------
function mumpsSolver(Ke::SparseMatrixCSC, rhs::AbstractArray; ooc::Int=0,
    saveFac::Bool=false)

    # the coefficient matrix is simply complex symmetric (not hermitan), so an LDLᴴ
    # decomposition can be used. MUMPS package is used for this purpose.

    sym  = 1
    @time Ainv = factorMUMPS(Ke, sym, ooc)

    nD = size(rhs, 1)
    ns = size(rhs, 2)
    xt  = zeros(Complex{Float64}, nD, ns)

    if maximum(abs.(rhs)) == 0.0
        println("All source terms are zeros.")
    else
        xt = applyMUMPS(Ainv, rhs)
    end

    saveFac && ( return xt, Ainv )

    destroyMUMPS(Ainv)
    return xt

end


#-------------------------------------------------------------------------------
"""
    `compRxProjectionTerm(mtMesh, mtData)`

"""
function compRxProjectionTerm(mtMesh::TensorMesh2D, mtData::MTData)

    #
    yLen = mtMesh.yLen
    zLen = mtMesh.zLen
    ny   = length(yLen)
    nz   = length(zLen)
    Gz   = kron(ddx(nz), spunit(ny+1))
    L2   = kron(sdiag(1 ./ zLen), spunit(ny+1))
    zGrad = L2 * Gz

    #
    yNode = cumsum([0; yLen]) .- mtMesh.origin[1]
    zNode = cumsum([0; zLen]) .- mtMesh.origin[2]
    nNode = (ny+1) * (nz+1)
    nEy   = ny * (nz+1)
    nEz   = (ny+1) * nz
    nEdge = nEy + nEz

    # interpolation matrix
    rxLoc = mtData.rxLoc
    nr = size(rxLoc, 1)
    nodeMat = spzeros(nNode, nr)
    edgeMat = spzeros(nEz, nr)
    #
    for k = 1:nr
        rxDiple = rxLoc[k:k, :]
        (inds, weights) = interpMat2D(rxDiple, yNode, zNode)
        idx = zeros(Int, 4)
        for i = 1:4
            idx[i] = (ny+1) * (inds[1][i][2]-1) + inds[1][i][1]
        end
        weights = vec(weights)
        nodeMat[:, k] = sparsevec(idx, weights, nNode)
    end

    #
    zCen = zNode[1:end-1] + diff(zNode) / 2
    for k = 1:nr
        rxDiple = rxLoc[k:k, :]
        (inds, weights) = interpMat2D(rxDiple, yNode, zCen)
        idx = zeros(Int, 4)
        for i = 1:4
            idx[i]  = (ny+1) * (inds[1][i][2]-1) + inds[1][i][1]
            # idx[i] += nEy
        end
        weights = vec(weights)
        edgeMat[:, k] = sparsevec(idx, weights, nEz)
    end

    rxMat = RxProjMat(zGrad, nodeMat, edgeMat)

    return rxMat

end
