#-------------------------------------------------------------------------------
#
# defines routines for mesh setup and operators for discretization
#
# (c) HB, 25 Dec., 2015
# (c) Peng Ronghua, Aug., 2020 for v0.7
#
#-------------------------------------------------------------------------------
export setupTensorMesh2D!
export getNodalGradient2D
export aveCell2Node2D, aveCell2Face2D
export meshGeoFace2D, meshGeoEdge2D, meshGeoEdgeInv2D
export getCellGradient2D
export spunit, spdiag, ddx, av, avcn, sdiag
#-------------------------------------------------------------------------------
function setupTensorMesh2D!(em2dMesh::TensorMesh2D)

    yLen = em2dMesh.yLen
    zLen = em2dMesh.zLen
    gridSize = em2dMesh.gridSize

    em2dMesh.Face  = meshGeoFace2D(yLen, zLen)
    em2dMesh.Grad  = getNodalGradient2D(yLen, zLen)
    em2dMesh.AveCN = aveCell2Node2D(gridSize)
    em2dMesh.AveCF = aveCell2Face2D(gridSize)
    em2dMesh.setup = true
end


#-------------------------------------------------------------------------------
"""
`getNodalGradient2D` gets 2D nodal gradient operator.

"""
function getNodalGradient2D(d1::Vector, d2::Vector)

    n1 = length(d1)
    n2 = length(d2)

    G1 = kron(spunit(n2+1), ddx(n1))
    G2 = kron(ddx(n2), spunit(n1+1))
    Grad = [G1; G2]

    L = meshGeoEdgeInv2D(d1, d2)
    Grad = L * Grad

    return Grad
end


#-------------------------------------------------------------------------------
function getCellGradient2D(d1::Vector, d2::Vector)

    n1 = length(d1)
    n2 = length(d2)

    G1 = kron(spunit(n2), ddx(n1-1))
    G2 = kron(ddx(n2-1), spunit(n1))
    Grad = [G1; G2]

    return Grad

end

function getCellGradient2D(mtMesh::TensorMesh2D)

    #
    yLen = mtMesh.yLen
    zLen = mtMesh.zLen
    Grad = getCellGradient2D(yLen, zLen)

    return Grad

end


#-------------------------------------------------------------------------------
"""
    meshGeoFace2D(xLen, yLen)

Compute face area matrix of the 2D grid

"""
function meshGeoFace2D(d1::Vector, d2::Vector)

    return kron(sdiag(d2), sdiag(d1))

end


# get edge length for a 2D mesh.
function meshGeoEdge2D(d1::Vector, d2::Vector)

    n1 = length(d1)
    n2 = length(d2)

    L1 = diag(kron(spunit(n2+1), sdiag(d1)))
    L2 = diag(kron(sdiag(d2), spunit(n1+1)))
    L = sdiag([L1; L2])

    return L
end

function meshGeoEdgeInv2D(d1::Vector, d2::Vector)

    n1 = length(d1)
    n2 = length(d2)

    L1 = kron(spunit(n2+1), sdiag(1 ./ d1))
    L2 = kron(sdiag(1 ./ d2), spunit(n1+1))
    L = blockdiag(L1, L2)

    return L

end

# averaging mapping from cell-center to node
function aveCell2Node2D(n::Vector)
    A1 = avcn(n[1])
    A2 = avcn(n[2])
    return kron(A2, A1)
end


# averaging mapping from cell-center to face
function aveCell2Face2D(n::Vector)
    A1 = kron(spunit(n[2]), avcn(n[1]))
    A2 = kron(avcn(n[2]), spunit(n[1]))
    return [A2; A1]
end

#------------------------------------------------------------------------------
"""
    spunit(n)

Construct sparse unit matrix, replace the built-in function speye that is deprecated.

"""
function spunit(n::Integer)
    return sparse(1.0I, n, n)
end


"""
    spdiag(d1, d2, (x1, x2), m, n)

"""
function spdiag((x1,x2), (d1,d2), m, n)
    I, J, V = SparseArrays.spdiagm_internal(d1 => x1, d2 => x2)
    return sparse(I, J, V, m, n)

end


"""
    ddx(n)

Form 1D difference sparse matrix from node to center

"""
function ddx(n::Integer)
    return spdiag((-ones(n), ones(n)), (0,1), n, n+1)
end


"""
    av(n)

Form 1D averaging matrix from node to cell-center

"""
function av(n::Integer)
    return spdiag((0.5*ones(n), 0.5*ones(n)), (0,1), n, n+1)
end


"""
    av(n)

Form 1D averaging matrix from from cell-center to node

"""
function avcn(n::Integer)

    avn = spdiag((0.5*ones(n), 0.5*ones(n)), (-1,0), n+1, n)
    avn[1,1]     = 1.0
    avn[end,end] = 1.0

    return avn
end


"""
    sdiag(v)

Form sparse diagonal matrix

"""
function sdiag(v::AbstractVector)
    return spdiagm(0 => v)
end
