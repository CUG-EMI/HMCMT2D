#-------------------------------------------------------------------------------
"""
    mt1DAnalyticField(freq, sigma, zNode; eTop, compH)

computes analytic fields for 1D layered model. Assuming a halfspace with the
same conductivity as the bottom layer, and e^{iwt} time dependence for Ex-Hy.
Note: length(sigma) = length(zNode)-1

 It first computes the impedance of the top layer to determine the values of
 up- and down-going E-field components of the top layer, and then propagates
 the field components to deeper layers. It checks for overflow due to large
 exp(ikh).

#Arguments:
 - Input:
    * `freq`  =::Float64, frequency
    * `sigma` =::Array, conductivity of layered model
    * `zNode` =::depth of top of each layer
    * `eTop`: the given top boundary E-field value
    * `compH`: indicates whether compute the H-field or not

"""
function mt1DAnalyticField(freq::T, sigma::Array{T}, zNode::Array{T},
                           compH::Bool=false) where {T<:Float64}

    # check layer's top and sigma are the same size
    if length(sigma) != length(zNode)-1
        error("layer's conductivity is not the same size with its depth.")
    end
	
	eTop = 1.0+0im
    
	# physical constant
    mu0   = 4 * pi * 1e-7
    eps0  = 8.85 * 1e-12
    omega = 2 * pi * freq
    omu0  = omega * mu0

    # assume halfspace at the bottom of model
    sigma = collect([sigma; sigma[end]])
    nLayer = length(zNode)
    zLen   = diff(zNode)

    # First, compute the impedance at the top layer using the well-known
    # recurrence formula for impedances.

    # wave number and intrinsic impedance of the bottom layer
    k = sqrt(mu0*eps0*omega^2 - mu0*sigma[end]*omega*1im)
    ztmp = omega * mu0 / k

    for j = nLayer-1:-1:1
        k = sqrt(mu0*eps0*omega^2 - mu0*sigma[j]*omega*1im)
        zp = omega * mu0 / k
        ztmp = zp * (ztmp + zp * tanh(k*zLen[j]*1im)) / (zp + ztmp * tanh(k*zLen[j]*1im))
    end
    z0 = ztmp

    # Up- and down-going electric fileds of each layer boundary
    eLayer = zeros(ComplexF64, 2, nLayer)

    # Up- and down-going components of the top layer, respectively
    eLayer[1, 1] = 0.5 * eTop * (1 - omega * mu0 / (z0*k))
    eLayer[2, 1] = 0.5 * eTop * (1 + omega * mu0 / (z0*k))

    # wave number of all layers
    ka = sqrt.(mu0*eps0*omega^2 .- mu0*sigma*omega*1im)

    # Second propagate the EM fields from top to bottom
    for i = 1:nLayer-1
        kr = ka[i] / ka[i+1]
        pInv = 0.5 * [1+kr 1-kr; 1-kr 1+kr]
        eUD = [exp(ka[i]*zLen[i]*1im) 0;
               0 exp(-ka[i]*zLen[i]*1im)]
        eLayer[:, i+1] = pInv * eUD * eLayer[:, i]

        # check if overflow happens
        e2 = abs(eLayer[1, i+1] + eLayer[2, i+1])
        e1 = abs(eLayer[1, i] + eLayer[2, i])
        if e2-e1>0 || isnan(e2)
            eLayer[:, i+1:end] .= 0.0
            break
        end
    end

    eField = transpose(sum(eLayer, dims=1))

    if compH
        hLayer = [ eLayer[1:1, :] * sparse(Diagonal(-ka)) /omu0;
                   eLayer[2:2, :] * sparse(Diagonal(ka))  /omu0 ]

        hField = transpose(sum(hLayer, dims=1))

        return eField, hField
    end

    return eField

end


#-------------------------------------------------------------------------------
"""
`mt1DImpedance(freqArray, sigma, zNode)` calculates the impedance at the earth's
 surface for MT 1D layered model.
 Note: length(sigma) = length(zNode)

Input:
    `freqArray` =::Array, frequencies
    `sigma`     =::Array, conductivity model
    `zNode`     =::Array, location of top of each layer

Output:
    z1d  =::Array, impedance at surface
"""
function mt1DImpedance(freqArray::Array{T}, sigma::Array{T},
                    zNode::Array{T}, isRho=false) where {T<:Float64}

    # physical constant
    mu0   = 4 * pi * 1e-7
    eps0  = 8.85 * 1e-12
    nFreq = length(freqArray)
    zLen  = diff(zNode)
    nLayer = length(zNode)

    # impedances at surface
    z1d = zeros(ComplexF64, nFreq)

    if isRho
        rho = zeros(nFreq)
        pha = zeros(nFreq)
    end

    for i = 1:nFreq

        omega = 2 * pi * freqArray[i]
        k = sqrt(mu0*eps0*omega^2 - mu0*sigma[end]*omega*1im)

        # calculate the impedance at the bottom layer
        ztmp = omega * mu0 / k

        for j = 2:nLayer

            ind = nLayer - j + 1
            k = sqrt(mu0*eps0*omega^2 - mu0*sigma[ind]*omega*1im)

            zp = omega * mu0 / k

            ztmp = zp * (ztmp + zp * tanh(k*zLen[ind]*1im)) / (zp + ztmp * tanh(k*zLen[ind]*1im))
        end
        z1d[i] = ztmp

        if isRho
            rho[i] = abs2(z1d[i]) / (omega * mu0)
            pha[i] = atan2(imag(z1d[i]), real(z1d[i])) * 180/pi
        end

    end

    if isRho
        return rho, pha
    else
        return z1d
    end

end


#-------------------------------------------------------------------------------
"""
`compMT1DEField(freq, sigma, zNode)` calculates MT electric field for 1D layered model.
  Note: length(sigma) = length(zNode)-1

Input:
    `freq`  =::Float64, frequency
    `sigma` =::Array, conductivity of layered model
    `zNode` =::depth of top of each layer

Output:
    `eField` =::Array, computed electric field

"""
function compMT1DEField(freq::T, sigma::Array{T}, zNode::Array{T}) where {T<:Float64}

    # get EM operators
    Grad = comp1dNodeGradient(zNode)

    # mass matrices
    mu0   = 4 * pi * 1e-7
    omega = 2 * pi * freq
    h = diff(zNode)
    n = length(h)

    MmuF  = spdiagm(h/mu0)
    AveCE = avcn(n)
    Msig  = spdiagm(AveCE * (h .* sigma))

    # set up coefficent matrix
    A = Grad' * MmuF * Grad + 1im * omega * Msig

    # define the inner part of the solution matrix
    Aii = A[2:end, 2:end]
    # define the outer part of the solution matrix
    Aio = A[2:end,[1,end]]

    # set the boundary conditions
    Et = mt1DAnalyticField(freq, sigma, zNode)

    # scale the fields to be equal to unit at the top
    Et = Et / Et[1]
    bc = [Et[1], Et[end]]

    # form rhs
    rhs = -Aio * bc

    Eii = Aii \ rhs

    #
    eField = vcat(bc[1],Eii,bc[2])
    return eField

end


#-------------------------------------------------------------------------------
"""
`comp1dNodeGradient(node)` gets 1D nodal gradient operator.

"""
function comp1dNodeGradient(node::Array{Float64})

    #
    D = diff(node)
    n = length(D)

    G = ddx(n)
    L = sdiag(1 ./ D)

    # ∇⋅ = (D_{i+1} - D_{i})/h
    G = L * G

    return G

end
