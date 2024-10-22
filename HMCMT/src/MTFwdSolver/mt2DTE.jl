#-------------------------------------------------------------------------------
"""
`compMT2DTE` computes TE responses for a 2D model for a single frequency.

Input:
    `freq:`    =::Float64, a single frequency value
    `mt2dMesh` =::TensorMesh2D
    'coeMat'   =::CoeffMat, coefficient matrix
    'rxLoc'    =::Array, receivers locations
    'dataType' =::String, the type of responses (impedance or rho & phase)
    `linearSolver`: a string to determine which linear solver is used

Output:
    `respTE` =::Array{Float64, 2}, responses at receiver locations
    `eField` =::Vector{ComplexF64}, the electric fields at grid nodes
    `Ainv`   =::MUMPSfactorization, system matrix decomposition factor

"""
function compMT2DTE(freq::T, mt2dMesh::TensorMesh2D, coeMat::CoeffMat,
                    rxLoc::Array{T,2}, dataType::String;
                    linearSolver::String="") where {T<:Float64}

    # pre-assign values to the outputs
    Ainv = MUMPSfactorization(0, 0, 0, 0.0, 0.0)

    yLen   = mt2dMesh.yLen
    zLen   = mt2dMesh.zLen
    origin = mt2dMesh.origin
    sigma  = mt2dMesh.sigma

    yNode = [0; cumsum(yLen)] .- origin[1]
    zNode = [0; cumsum(zLen)] .- origin[2]

    ny = length(yLen)
    nz = length(zLen)

    omega = 2 * pi * freq
    Aii = coeMat.rAii + 1im * omega * coeMat.iAii
    Aio = coeMat.rAio + 1im * omega * coeMat.iAio

    # compute boundary fields
    bc = getBoundaryMT2DTE(freq, yLen, zLen, sigma)
    # set up RHS
    rhs = -Aio * bc

    # solve the linear system
    if isempty(linearSolver)
        Ainv = lu(Aii)
        Eii  = collect(Ainv) \ rhs
    elseif lowercase(linearSolver) == "mumps"
        sym  = 1
        Ainv = factorMUMPS(Aii, sym)
        Eii = applyMUMPS(Ainv, rhs)
        # destroyMUMPS(Ainv)
    end

    E2d = zeros(ComplexF64, nz+1, ny+1)
    E2d[1,:]         = bc[1:ny+1]
    E2d[2:end,1]     = bc[ny+2:ny+nz+1]
    E2d[2:end,end]   = bc[ny+nz+2:ny+2*nz+1]
    E2d[end,2:end-1] = bc[ny+2*nz+2:end]
    E2d[2:end-1,2:end-1] = copy(transpose(reshape(Eii, ny-1, nz-1)))

    # find out on the top of which layer the receivers are located.
    # assume all reveivers have the same z-loc (no topography).
    zRx = rxLoc[1,2]
    zid = findfirst(x -> x<0.1, abs.(zNode .- zRx))

    # extract things needed for interpolation
    Er01   = copy(transpose(E2d[zid:zid+1, :]))
    sigma1 = sigma[(zid-1)*ny+1:zid*ny]
    zLen1  = zLen[zid]

    # save the solution (i.e. the electric fields at grid nodes) for later use.
    eField = vec(copy(transpose(E2d)))

    (Exr, Hyr) = compFieldsAtRxTE(omega, rxLoc, yNode, zLen1, sigma1, Er01)

    respTE = compMTRespTE(omega, Exr, Hyr, dataType)

    return respTE, eField, Ainv

end


#-------------------------------------------------------------------------------
"""
`getBoundaryMT2DTE` computes boundary E fields.

Input:
    `freq:` =::Float64, a single frequency value
    `yLen`  =::Vector, cell sizes in y direction
    `zLen`  =::Vector, cell sizes in z direction
    `sigma` =::Vector, cell conductivity

Output:
    `bc` =::Array{ComplexF64, 1}, boundary E fields

"""
function getBoundaryMT2DTE(freq::T, yLen::Vector{T}, zLen::Vector{T},
                        sigma::Vector{T}) where {T<:Float64}

    ny = length(yLen)
    nz = length(zLen)
    zNode = [0; cumsum(zLen)]
    sigma2D = (reshape(sigma, ny, nz))'

    nb = 2*(ny+nz)
    bc = zeros(ComplexF64, nb)

    # top boundary
    bc[1:ny+1] .= 1.0+0.0im

    # left boundary
    sigma1D = sigma2D[:,1]
    eb = mt1DAnalyticField(freq, sigma1D, zNode)
    eb = eb / eb[1]
    bc[ny+2:ny+nz+1] = eb[2:end]

    # right boundary
    sigma1D = sigma2D[:,end]
    eb = mt1DAnalyticField(freq, sigma1D, zNode)
    eb = eb / eb[1]
    bc[ny+nz+2:ny+2*nz+1] = eb[2:end]

    # bottom boundary
    for i=2:ny
        sigma1D = (sigma2D[:,i-1]*yLen[i-1] + sigma2D[:,i]*yLen[i])/(yLen[i-1] + yLen[i])
        eb = mt1DAnalyticField(freq, sigma1D, zNode)
        bc[ny+2*nz+i] = eb[end] / eb[1]
    end

    return bc
end


#-------------------------------------------------------------------------------
"""
`compFieldsAtRxTE` computes EM fields at receiver locations from fields at grid nodes

Input:
    `omega:` =::Float64, a single angular frequency value
    'rxLoc'  =::Array, receivers locations
    `yNode`  =::Vector, node y-coordinates
    `zLen1`  =::Float64, z-cell size of the receiver layer
    `sigma1` =::Vector, cell conductivity of the receiver layer
    `Er01`   =::Array{ComplexF64,2}, the two row grid node Ex
Output:
    `Exr` =::Vector{ComplexF64}, Ex at the receiver locations
    `Hyr` =::Vector{ComplexF64}, Hy at the receiver locations

"""
function compFieldsAtRxTE(omega::T, rxLoc::Array{T,2}, yNode::Vector{T},
        zLen1::T, sigma1::Vector{T}, Er01::Array{ComplexF64,2}) where {T<:Float64}

    yLen = diff(yNode)
    ny   = length(yLen)
    mu0 = 4 * pi * 1e-7
    mu  = mu0 * ones(ny)

    # Ex & Hy at receiver locations
    nRx = size(rxLoc,1)
    Exr = zeros(ComplexF64,nRx)
    Hyr = copy(Exr)

    # First compute fields at the receiver layer (earth surface or seafloor).
    Ex0 = Er01[:, 1]

    Bz0 = (ddx(ny) * Er01[:,1]) ./ yLen / (1im*omega)
    Bz1 = (ddx(ny) * Er01[:,2]) ./ yLen / (1im*omega)

    # quarter Hz (1/4), with length ny
    HzQ = (0.75 * Bz0 + 0.25 * Bz1) ./ mu

    # half Hy (1/2), with length ny-1
    # More strictly, an average mu should be used here.
    HyH = - (Er01[2:end-1,2] - Er01[2:end-1,1]) / zLen1 / (1im*omega*mu0)

    # quarter Ex (1/4), with length ny-1
    ExQ = 0.75 * Er01[2:end-1,1] + 0.25 * Er01[2:end-1,2]

    # average conductiviy at vertical edge
    sigma1v = (av(ny-1)*(sigma1 .* yLen)) ./ (av(ny-1)*yLen)

    # dHz/dy
    dHzQ = (ddx(ny-1)*HzQ) ./ (av(ny-1)*yLen)

    # Ampre's theorem: dHz/dy - dHy/dz = sigma1v.*ExQ,
    # where dHy/dz = (HyH-Hy0)/(0.5*zLen1).
    Hy0 = zeros(ComplexF64,ny+1)
    Hy0[2:end-1] = HyH - (dHzQ - sigma1v .* ExQ) * (0.5*zLen1)
    Hy0[1]   = Hy0[2]
    Hy0[end] = Hy0[end-1]

    # Second interpolate fields to receiver locations (using linear interpolation)
    for ir=1:size(rxLoc,1)
        rxY = rxLoc[ir,1]
        id = findfirst(x -> x>rxY, yNode)
        if id==0
            error("The receiver location seems to be out of range!")
        end

        dy1 = rxY - yNode[id-1]
        dy2 = yNode[id] - rxY
        Exr[ir] = Ex0[id-1]*dy2 + Ex0[id]*dy1
        Hyr[ir] = Hy0[id-1]*dy2 + Hy0[id]*dy1
    end

    return Exr, Hyr
end


#-------------------------------------------------------------------------------
function compFieldsAtRxTE(omega::Float64, exTE::Array{ComplexF64}, rxMat::RxProjMat)

    # first compute magnetic field
    mu0 = 4 * pi * 1e-7
    hyTE = - rxMat.zGrad * exTE / (1im*omega*mu0)

    # get electric and magnetic fields at receiver locations
    Exr = rxMat.nodeMat' * exTE
    Hyr = rxMat.edgeMat' * hyTE

    return Exr, Hyr

end

#-------------------------------------------------------------------------------
"""
`compMTRespTE` MT TE responses from EM fields.

Input:
    `omega:` =::Float64, a single angular frequency value
    `Ex`     =::Vector{ComplexF64}
    `Hy`     =::Vector{ComplexF64}
    'dataType' =::String, the type of responses (impedance or rho & phase)
Output:
    `respTE` =::Array{Float64, 2}
"""
function compMTRespTE(omega::Float64, Ex::Vector{T}, Hy::Vector{T},
                    dataType::String) where {T<:ComplexF64}

    mu0 = 4 * pi * 1e-7
    respTE = zeros(Float64,length(Ex),2)

    Zxy = Ex ./ Hy

    if occursin("Impedance", dataType)
        respTE = [real(Zxy) imag(Zxy)]
        return respTE
    end

    rho = (abs.(Zxy)).^2 / (omega*mu0)
    phs = atan.(imag(Zxy), real(Zxy)) * 180/pi
    respTE = [rho phs]

    return respTE

end
