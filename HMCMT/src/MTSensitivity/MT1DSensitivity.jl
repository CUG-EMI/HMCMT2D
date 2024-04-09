export mt1DFieldSensMatrix, compImpJacMatrix
export getBCDerivMatrix, getBCderivTE, getBCderivTM

#-------------------------------------------------------------------------------
"""
    `mt1DFieldSensMatrix(freq, sigma, zNode, source, fTop)`

Computes analytic fields and their sensitivities for 1D layered model.
It assumes: (1) time dependence: e^{iwt}; (2) wavenumber: k^2 = -iwu/p + uew^2

* Input:
     -freq   :  a single frequency value
     -sigma  :  1D array, conductivity of layered model
     -zNode  :  1D array, depth of top of each layer
     -source :  string, 'E' or 'H', determin the top E- or H-field is source.
     -fTop   :  the given top boundary field value

* Output:
     -eField  : 1D array, electric fields at the top of each layer
     -hField  : 1D array, magnetic fields at the top of each layer
     -dE      : 2D array, electric field sensitivity
     -dH      : 2D array, magnetic field sensitivity

"""
function mt1DFieldSensMatrix(freq::T, sig1d::Array{T}, zNode::Array{T},
    source::String="E", fTop::T=1.0) where {T<:Float64}

    #
    # check layer's top and sigma are the same size
    if length(sig1d) != length(zNode)-1
        error("layer's conductivity is not the same size with its depth.")
    end

    # physical constant
    mu0   = 4 * pi * 1e-7
    eps0  = 8.85 * 1e-12
    omega = 2 * pi * freq
    omu   = omega * mu0

    sigma = copy(sig1d)
    # assume halfspace at the bottom of model
    append!(sigma, sigma[end])
    nLayer = length(sigma)
    zLen   = diff(zNode)

    # first: compute the impedance at the top layer and its derivative
    (z1, dz1) = compImpJacMatrix(freq, sigma, zLen)

    # up- and down-going electric and magnetic fileds of each layer boundary
    eLayer = zeros(ComplexF64, 2, nLayer)
    dEu    = zeros(ComplexF64, nLayer, nLayer)
    dEd    = zeros(ComplexF64, nLayer, nLayer)

    hLayer = zeros(ComplexF64, 2, nLayer)
    dHu    = zeros(ComplexF64, nLayer, nLayer)
    dHd    = zeros(ComplexF64, nLayer, nLayer)

    # wavenumber of all layers and its derivatives
    ka = sqrt.(-1im * omu * sigma)
    dkaVec = (-1im *omu / 2) ./ ka
    dka = sdiag(dkaVec)

    # up- and down-going components of the top layer, respectively
    k1 = ka[1]

    if source == "E"          # E1 = c
        eLayer[1,1] = 0.5 * fTop * (1 - omu / (z1*k1))
        eLayer[2,1] = 0.5 * fTop * (1 + omu / (z1*k1))

        hLayer[1,1] = -k1/omu * eLayer[1,1]
        hLayer[2,1] =  k1/omu * eLayer[2,1]

        dEu[1, :] = 0.5 * fTop * omu / (z1*k1) * (1/z1*dz1 + 1/k1*dka[1, :])
        dEd[1, :] = - dEu[1, :]    # because Eu1+Ed1 = constant

        dHu[1, :] = -eLayer[1,1]/omu * dka[1, :] - ka[1]/omu * dEu[1, :]
        dHd[1, :] =  eLayer[2,1]/omu * dka[1, :] + ka[1]/omu * dEd[1, :]

    elseif source == "H"     # H1 = c
        hLayer[1,1] = 0.5 * fTop * (1 - z1*k1 / omu)
        hLayer[2,1] = 0.5 * fTop * (1 + z1*k1 / omu)

        eLayer[1,1] = -omu/k1 * hLayer[1,1]
        eLayer[2,1] =  omu/k1 * hLayer[2,1]

        dHu[1, :] = -0.5 * fTop/omu * (z1*dka[1, :] + k1*dz1)
        dHd[1, :] = - dHu[1, :]    # because Hu1+Hd1 = constant

        dEu[1, :] = 0.5 * fTop * (dz1 + (omu/k1^2)*dka[1, :])
        dEd[1, :] = 0.5 * fTop * (dz1 - (omu/k1^2)*dka[1, :])

    end

    # the exponential term and its derivatives.
    expt  = exp.(1im * ka[1:end-1] .* zLen)     # length: nLayer-1
    expr  = 1 ./ expt
    dexpt =  1im * zLen .* expt .* dkaVec[1:end-1]
    dexpr = -1im * zLen .* expr .* dkaVec[1:end-1]
    dexpt = hcat( sdiag(dexpt), spzeros(nLayer-1, 1) )  # size: (nLayer-1, nLayer)
    dexpr = hcat( sdiag(dexpr), spzeros(nLayer-1, 1) )

    # the wavenumber ratio term and its derivatives.
    kr  = ka[1:end-1] ./ ka[2:end]        # length: nLayer-1
    dkr = zeros(ComplexF64, nLayer-1, nLayer)
    for j=1:nLayer-1
        dkr[j, :] = dka[j,:] / ka[j+1] - ka[j]/(ka[j+1]^2) * dka[j+1,:]
    end

    # the four mix terms and their derivatives.
    mix11 = (1 .+ kr) .* expt               # length: nLayer-1
    mix12 = (1 .- kr) .* expr
    mix21 = (1 .- kr) .* expt
    mix22 = (1 .+ kr) .* expr
    dmix11 = zeros(ComplexF64, nLayer-1, nLayer)
    dmix12 = zeros(ComplexF64, nLayer-1, nLayer)
    dmix21 = zeros(ComplexF64, nLayer-1, nLayer)
    dmix22 = zeros(ComplexF64, nLayer-1, nLayer)
    for j=1:nLayer-1
        dmix11[j, :] = (1 + kr[j] ) * dexpt[j, :] + expt[j] * dkr[j, :]
        dmix12[j, :] = (1 - kr[j] ) * dexpr[j, :] - expr[j] * dkr[j, :]
        dmix21[j, :] = (1 - kr[j] ) * dexpt[j, :] - expt[j] * dkr[j, :]
        dmix22[j, :] = (1 + kr[j] ) * dexpr[j, :] + expr[j] * dkr[j, :]
    end

    # second: propagate the EM fields from top to bottom
    for j = 1:nLayer-1
        pInv = 0.5 * [1+kr[j] 1-kr[j]; 1-kr[j] 1+kr[j]]
        eUD = [expt[j] 0; 0 expr[j]]
        eLayer[:, j+1] = pInv * eUD * eLayer[:, j]

        eu = eLayer[1, j]
        ed = eLayer[2, j]

        dEu[j+1, :] = 0.5*( dmix11[j,:] * eu + mix11[j] * dEu[j, :] +
                            dmix12[j,:] * ed + mix12[j] * dEd[j, :] )

        dEd[j+1, :] = 0.5*( dmix21[j,:] * eu + mix21[j] * dEu[j, :] +
                            dmix22[j,:] * ed + mix22[j] * dEd[j, :] )

        epu = eLayer[1, j+1]
        epd = eLayer[2, j+1]
        dHu[j+1, :] = -epu/omu * dka[j+1, :] - ka[j+1] / omu * dEu[j+1, :]
        dHd[j+1, :] =  epd/omu * dka[j+1, :] + ka[j+1] / omu * dEd[j+1, :]

        # check if overflow happens
        e2 = abs(eLayer[1, j+1] + eLayer[2, j+1])
        e1 = abs(eLayer[1, j] + eLayer[2, j])
        if e2-e1 > 0.0 || isnan(e2)
            eLayer[:, j+1:end]    .= 0.0
            dEu[j+1:end, j+1:end] .= 0.0
            dEd[j+1:end, j+1:end] .= 0.0
            dHu[j+1:end, j+1:end] .= 0.0
            dHd[j+1:end, j+1:end] .= 0.0
            break
        end

    end

    dE = dEu + dEd
    dH = dHu + dHd

    # drop the derivative with respect to the bottom sigma
    dE = dE[1:end, 1:end-1]
    dH = dH[1:end, 1:end-1]

    eField = copy(transpose(sum(eLayer, dims=1)))

    if source == "E"
        return eField, dE
    elseif source == "H"
        hLayer = [eLayer[1:1,:] * sdiag(-ka) /omu; eLayer[2:2,:] * sdiag(ka)  /omu ]
        hField = copy(transpose(sum(hLayer, dims=1)))
        return hField, dH
    end

end


#-------------------------------------------------------------------------------
"""
    `compImpJacMatrix(freq, sig1d, thick1d)`

Computes the derivative of impedance (Z) with respect to layer conductivities

for MT 1-D isotropic problem using analytical method with a time dependence of e^{iwt}

"""
function compImpJacMatrix(freq::T, sig1d::Vector{T}, thick1d::Vector{T}) where {T<:Float64}

    # check layer's top and sigma are the same size
    nLayer = length(sig1d)
    mu0    = 4 * pi * 1e-7

    omega = 2 * pi * freq
    iom   = 1im * omega * mu0

    #
    Z        = 0.0 + 0.0 * 1im
    dZ_ZP1   = zeros(ComplexF64, nLayer)
    dZ_sigma = zeros(ComplexF64, nLayer)
    zimpDeri = zeros(ComplexF64, nLayer)

    # Loop over layers
    for j = nLayer:-1:1
        k = sqrt(-iom * sig1d[j])
        Zt  = omega * mu0 / k                     # intrinsic impedance of this layer
        dZt = 1im * (omega * mu0)^2 / (2*k^3)     # dZt/dsigma

        if j == nLayer
            Z = Zt
            dZ_sigma[j] = dZt
            continue
        end

        RI     = (Zt-Z) / (Zt+Z)
        theEXP = exp(-2im * k * thick1d[j] )
        L      = RI * theEXP
        Ztmp   = Zt * (1-L)/(1+L)

        # dL/dsigma
        dL = 2*Z/(Zt+Z)^2 * theEXP * dZt + (-2im * thick1d[j] * L) * (-iom/2/k)

        dZ_ZP1[j] = 4 * Zt * Zt * theEXP / ((1+L) * (Zt+Z) )^2
        dZ_sigma[j] = dZt * (1-L) / (1+L) + Zt * (-2) / (1+L)^2 * dL

        Z = Ztmp

    end

    # Loop over layers, chain rule
    for iLayer = nLayer:-1:2
        dZ_ZPN = 1
        for j = 1:iLayer-1
            dZ_ZPN = dZ_ZPN * dZ_ZP1[j]
        end
        zimpDeri[iLayer] = dZ_ZPN * dZ_sigma[iLayer]
    end

    zimpDeri[1] = dZ_sigma[1]

    return Z, zimpDeri

end


#-------------------------------------------------------------------------------
"""
    `getBCDerivMatrix(mtMesh, freq, sigma, source)`

Computes the derivative of boundary Ex fields for TE mode OR Hx fields for TM mode.

"""
function getBCDerivMatrix(freq::T, yLen::Vector{T}, zLen::Vector{T},
                          sigma::Vector{T}, source::String) where {T<:Float64}

    #
    ny = length(yLen)
    nz = length(zLen)
    zNode = [0.0; cumsum(zLen)]
    sig2D = copy(transpose(reshape(sigma, ny, nz)))

    # number of boundary E fields
    nb    = 2*(ny+nz)
    ncell = ny*nz
    bc  = zeros(ComplexF64, nb)
    dBC = zeros(ComplexF64, nb, ncell)

    # top boundary and its derivative
    bc[1:ny+1] .= 1.0+0.0im
    #dBC[1:ny+1, :] = 0.0

    # left boundary
    idx   = collect(ny+2:ny+nz+1)
    sig1D = sig2D[:, 1]
    (eb, dE) = mt1DFieldSensMatrix(freq, sig1D, zNode, source, 1.0)
    bc[idx]  = eb[2:end]
    dBC[idx, 1:ny:end] = dE[2:end, :]

    # right boundary
    idx   = collect(ny+nz+2:ny+2*nz+1)
    sig1D = sig2D[:,end]
    (eb, dE) = mt1DFieldSensMatrix(freq, sig1D, zNode, source, 1.0)
    bc[idx]  = eb[2:end]
    dBC[idx, ny:ny:end] = dE[2:end, :]

    #  bottom boundary
    #=
    mtmp = spunit(ncell)
    for j=2:ny
        yLen1 = yLen[j-1];
        yLen2 = yLen[j];
        sig1D = (sig2D[:, j-1] * yLen1 + sig2D[:, j] * yLen2) / (yLen1 + yLen2);
        id1 = collect(j-1:ny:ncell);
        id2 = collect(j:ny:ncell);
        # dsigma1D/dsigma
        # dsig = (mtmp[id1,:]*yLen1 + mtmp[id2,:]*yLen2) / (yLen1 + yLen2)
        nidx1 = length(id1);
        nidx2 = length(id2);
        ii = vcat(collect(1:nidx1), collect(1:nidx2));
        jj = vcat(id1, id2);
        vv = vcat(ones(nidx1)*yLen1, ones(nidx2)*yLen2);
        dsig = sparse(ii, jj, vv, nidx1, ncell) / (yLen1 + yLen2);

        (eb, dE) = mt1DFieldSensMatrix(freq, sig1D, zNode, source, 1.0);

    	bc[ny+2*nz+j]     = eb[end]
        dBC[ny+2*nz+j, :] = dE[end:end, :] * dsig

    end
    =#

    # modified at Augest, 2022
    sig1D = vec(mean(sig2D,dims=2))
    (eb, dE) = mt1DFieldSensMatrix(freq, sig1D, zNode, source, 1.0);
    for j = 2:ny
        yLen1 = yLen[j-1];
        yLen2 = yLen[j];
        id1 = collect(j-1:ny:ncell);
        id2 = collect(j:ny:ncell);
        nidx1 = length(id1);
        nidx2 = length(id2);
        ii = vcat(collect(1:nidx1), collect(1:nidx2));
        jj = vcat(id1, id2);
        vv = vcat(ones(nidx1)*yLen1, ones(nidx2)*yLen2);
        dsig = sparse(ii, jj, vv, nidx1, ncell) / (yLen1 + yLen2);
        bc[ny+2*nz+j]     = eb[end]
        dBC[ny+2*nz+j, :] = dE[end:end, :] * dsig
    end
    #

    return dBC, bc

end

#-------------------------------------------------------------------------------
function getBCderivTE(freq::T, yLen::Vector{T}, zLen::Vector{T},
    sigma::Vector{T}) where {T<:Float64}

    #
    source = "E"
    (dBC, bc) = getBCDerivMatrix(freq, yLen, zLen, sigma, source)

    return dBC, bc

end

#-------------------------------------------------------------------------------
function getBCderivTM(freq::T, yLen::Vector{T}, zLen::Vector{T},
    sigma::Vector{T}) where {T<:Float64}

    #
    source = "H"
    (dBC, bc) = getBCDerivMatrix(freq, yLen, zLen, sigma, source)

    return dBC, bc

end
