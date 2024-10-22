"""
`compMT2DTM` computes TM responses for a 2D model for a single frequency.

Input:
    `freq:`    =::Float64, a single frequency value
    `mt2dMesh` =::TensorMesh2D
    'coeMat'   =::CoeffMat, coefficient matrix
    'rxLoc'    =::Array, receivers locations
    'dataType' =::String, the type of responses (impedance or rho & phase)
    `linearSolver`: a string to determine which linear solver is used

Output:
    `respTM` =::Array{Float64, 2}, responses at receiver locations
    `hField` =::Vector{ComplexF64}, the magnetic fields at grid nodes
    `Ainv`   =::MUMPSfactorization, system matrix decomposition factor

"""
function compMT2DTM(freq::T, mt2dMesh::TensorMesh2D, coeMat::CoeffMat,
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
    bc = getBoundaryMT2DTM(freq, yLen, zLen, sigma)
    # set up RHS
    rhs = -Aio * bc

    # solve the linear system
    if isempty(linearSolver)
        Ainv = lu(Aii)
        Hii  = collect(Ainv) \ rhs
    elseif lowercase(linearSolver) == "mumps"
        sym  = 1
        Ainv = factorMUMPS(Aii, sym)
        Hii = applyMUMPS(Ainv, rhs)
        # destroyMUMPS(Ainv)
    end

    H2d = zeros(ComplexF64, nz+1, ny+1)
    H2d[1,:]         = bc[1:ny+1]
    H2d[2:end,1]     = bc[ny+2:ny+nz+1]
    H2d[2:end,end]   = bc[ny+nz+2:ny+2*nz+1]
    H2d[end,2:end-1] = bc[ny+2*nz+2:end]
    H2d[2:end-1,2:end-1] = copy(transpose(reshape(Hii, ny-1, nz-1)))

    # find out on the top of which layer the receivers are located.
    # assume all reveivers have the same z-loc (no topography).
    zRx = rxLoc[1,2]
    zid = findfirst(x -> x<0.1, abs.(zNode .- zRx))

    # extract things needed for interpolation
    Hr01   = copy(transpose(H2d[zid:zid+1, :]))
    sigma1 = sigma[(zid-1)*ny+1:zid*ny]
    zLen1  = zLen[zid]

    Eyr, Hxr = compFieldsAtRxTM(omega, rxLoc, yNode, zLen1, sigma1, Hr01)

    respTM = compMTRespTM(omega, Eyr, Hxr, dataType)

    # save the solution (i.e. the magnetic fields at grid nodes) for later use.
    hField = vec(copy(transpose(H2d)))

    return respTM, hField, Ainv

    return
end



"""
`getBoundaryMT2DTM` computes boundary H fields.

Input:
    `freq:` =::Float64, a single frequency value
    `yLen`  =::Vector, cell sizes in y direction
    `zLen`  =::Vector, cell sizes in z direction
    `sigma` =::Vector, cell conductivity

Output:
    `bc` =::Array{ComplexF64, 1}, boundary H fields

"""
function getBoundaryMT2DTM(freq::T, yLen::Vector{T}, zLen::Vector{T},
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
    eb, hb = mt1DAnalyticField(freq, sigma1D, zNode, true)
    hb = hb / hb[1]
    bc[ny+2:ny+nz+1] = hb[2:end]

    # right boundary
    sigma1D = sigma2D[:,end]
    eb, hb = mt1DAnalyticField(freq, sigma1D, zNode, true)
    hb = hb / hb[1]
    bc[ny+nz+2:ny+2*nz+1] = hb[2:end]

    # bottom boundary
    for i=2:ny
        sigma1D = (sigma2D[:,i-1]*yLen[i-1] + sigma2D[:,i]*yLen[i]) /(yLen[i-1] + yLen[i])
        eb, hb = mt1DAnalyticField(freq, sigma1D, zNode, true)
        bc[ny+2*nz+i] = hb[end] / hb[1]
    end

    return bc
end


"""
`compFieldsAtRxTM` computes EM fields at receiver locations from fields at grid nodes

Input:
    `omega:` =::Float64, a single angular frequency value
    'rxLoc'  =::Array, receivers locations
    `yNode`  =::Vector, node y-coordinates
    `zLen1`  =::Float64, z-cell size of the receiver layer
    `sigma1` =::Vector, cell conductivity of the receiver layer
    `Hr01`   =::Array{ComplexF64,2}, the two row grid node Hx
Output:
    `Eyr` =::Vector{ComplexF64}, Ey at the receiver locations
    `Hxr` =::Vector{ComplexF64}, Hx at the receiver locations

"""
function compFieldsAtRxTM(omega::T, rxLoc::Array{T,2}, yNode::Vector{T},
            zLen1::T, sigma1::Vector{T}, Hr01::Array{ComplexF64,2}) where {T<:Float64}

    yLen = diff(yNode)
    ny   = length(yLen)
    mu0 = 4 * pi * 1e-7
    mu  = mu0 * ones(ny)

    # Ex & Hy at receiver locations
    nRx = size(rxLoc,1)
    Eyr = zeros(ComplexF64,nRx)
    Hxr = copy(Eyr)

    # First compute fields at the receiver layer (earth surface or seafloor).
    Hx0 = Hr01[:,1]

    Jz0 = -(ddx(ny) * Hr01[:,1]) ./ yLen
    Jz1 = -(ddx(ny) * Hr01[:,2]) ./ yLen

    # quarter Ez (1/4), with length ny
    EzQ = (0.75 * Jz0 + 0.25 * Jz1) ./ sigma1

    # half Ey (1/2), with length ny-1
    JyH   = (Hr01[2:end-1,2] - Hr01[2:end-1,1]) / zLen1
    rho1v = (av(ny-1)*((1 ./ sigma1) .* yLen)) ./ (av(ny-1)*yLen)
    EyH   = JyH .* rho1v

    # quarter Hx (1/4), with length ny-1
    HxQ = 0.75 * Hr01[2:end-1,1] + 0.25 * Hr01[2:end-1,2]

    # average permeability at vertical edge
    # muv = (avnc(ny-1)*(mu.*yLen)) ./ (avnc(ny-1)*yLen)

    # dEz/dy
    dEzQ = (ddx(ny-1)*EzQ) ./ (av(ny-1)*yLen)

    # Faraday's law: -dEz/dy + dEy/dz = 1im*omega*muv.*HxQ,
    # where dEy/dz = (EyH-Ey0)/(0.5*zLen1).
    Ey0 = zeros(ComplexF64,ny+1)
    Ey0[2:end-1] = EyH - (dEzQ + 1im*omega*mu0*HxQ)*(0.5*zLen1)
    Ey0[1]   = Ey0[2]
    Ey0[end] = Ey0[end-1]

    # Second interpolate fields to receiver locations (using linear interpolation)
    for ir=1:size(rxLoc,1)
        rxY = rxLoc[ir,1]
        id = findfirst(x -> x>rxY, yNode)
        if id==0
            error("The receiver location seems to be out of range!")
        end

        dy1 = rxY - yNode[id-1]
        dy2 = yNode[id] - rxY
        Eyr[ir] = Ey0[id-1]*dy2 + Ey0[id]*dy1
        Hxr[ir] = Hx0[id-1]*dy2 + Hx0[id]*dy1
    end

    return Eyr, Hxr
end


"""
`compMTRespTM` MT TM responses from EM fields.

Input:
    `omega:` =::Float64, a single angular frequency value
    `Ey`     =::Vector{ComplexF64}
    `Hx`     =::Vector{ComplexF64}
    'dataType' =::String, the type of responses (impedance or rho & phase)
Output:
    `respTM` =::Array{Float64, 2}
"""
function compMTRespTM(omega::Float64, Ey::Vector{T}, Hx::Vector{T},
                    dataType::String) where {T<:ComplexF64}

    mu0 = 4 * pi * 1e-7
    respTM = zeros(Float64,length(Ey),2)
    Zyx = Ey ./ Hx

    if occursin("Impedance", dataType)
        respTM = [real(Zyx) imag(Zyx)]
        return respTM
    end

    rho = (abs.(Zyx)).^2 / (omega*mu0)
    phs = atan.(imag(Zyx), real(Zyx)) * 180/pi
    respTM = [rho phs]

    return respTM

end
