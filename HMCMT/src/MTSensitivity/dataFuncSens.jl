export getDataFuncSensTE, getDataFuncSensTM

#-------------------------------------------------------------------------------
"""
`getDataFuncSensTE` computes partial derivative of MT data of a single frequency
  of TE mode with respect to solution fields (L) and model conductivities (Q).
  It is closely related to `compFieldsAtRxTE`.

Input:
    omega       :: Float64    - a single angular frequency value.
    rxSensInfo  :: PreRxSens  - receiver locations.
    Ex01        :: Array      - the two row grid node Ex.
    dataType    :: String
    dataComp    :: Array{String}

Output:
    L :: Array
    Q :: Array

"""
function getDataFuncSensTE(omega::Float64, rxSensInfo::PreRxSens,
    Ex01::Array{ComplexF64}, dataType::String, dataComp::Array{String})

    #------- 1st, compute the partial derivative of EM fields at sites -------#

    # extract things
    dEx0      = rxSensInfo.dFn0
    dEx1      = rxSensInfo.dFn1
    sigma1    = rxSensInfo.sigma1
    dsigma1   = rxSensInfo.dsigma1
    yLen      = rxSensInfo.yLen
    zLen1     = rxSensInfo.zLen1
    linRxMap  = rxSensInfo.linRxMap
    linRxMap2 = rxSensInfo.linRxMap2

    ny    = length(yLen)
    nNode = size(dEx0, 2)
    nCell = size(dsigma1, 2)

    MU0 = 4 * pi * 1e-7
    mu = MU0 * ones(ny)


    # First compute fields at the receiver layer (earth surface or seafloor).
    Bz0 = (ddx(ny) * Ex01[:,1]) ./yLen / (1im*omega)
    Bz1 = (ddx(ny) * Ex01[:,2]) ./yLen / (1im*omega)
    dtmp = sdiag(1 ./ yLen / (1im*omega)) * ddx(ny)
    dBz0 = dtmp * dEx0
    dBz1 = dtmp * dEx1

    # quarter Hz (1/4), with length ny
    HzQ = (0.75 * Bz0 + 0.25 * Bz1) ./ mu
    dHzQ = sdiag(1 ./ mu) * (0.75*dBz0 + 0.25*dBz1)

    # half Hy (1/2), with length ny-1
    # More strictly, an average mu should be used here.
    HyH = - (Ex01[2:end-1,2] - Ex01[2:end-1,1]) / zLen1 / (1im*omega*MU0)
    dHyH = - (dEx1[2:end-1,:] - dEx0[2:end-1,:]) / zLen1 / (1im*omega*MU0)

    # quarter Ex (1/4), with length ny-1
    ExQ = 0.75 * Ex01[2:end-1,1] + 0.25 * Ex01[2:end-1,2]
    dExQ = 0.75 * dEx0[2:end-1,:] + 0.25 * dEx1[2:end-1,:]


    # average conductiviy at vertical edge
    sigma1v = (avnc(ny-1)*(sigma1.*yLen)) ./ (avnc(ny-1)*yLen)
    dsigma1v = sdiag(1 ./ (avnc(ny-1)*yLen)) * avnc(ny-1) * sdiag(yLen) * dsigma1

    # dHz/dy
    dHzQ_dy = (ddx(ny-1)*HzQ) ./ (avnc(ny-1)*yLen)
    ddHzQ = sdiag(1 ./ (avnc(ny-1)*yLen)) * ddx(ny-1) * dHzQ

    # Ampre's theorem: dHz/dy - dHy/dz = sigma1v.*ExQ,
    # where dHy/dz = (HyH-Hy0)/(0.5*zLen1).
    Hy0 = zeros(ComplexF64, ny+1)
    Hy0[2:end-1] = HyH - (dHzQ_dy - sigma1v.*ExQ)*(0.5*zLen1)
    Hy0[1]   = Hy0[2]
    Hy0[end] = Hy0[end-1]

    dHy0 = spzeros(ComplexF64, ny+1, nNode)
    dHy0[2:end-1, :] = dHyH - (ddHzQ - sdiag(sigma1v)*dExQ)*(0.5*zLen1)
    dHy0[1, :] = dHy0[2, :]
    dHy0[end, :] = dHy0[end-1, :]

    dHy0_dsig = spzeros(ComplexF64, ny+1, nCell)
    dHy0_dsig[2:end-1, :] = 0.5*zLen1 * sdiag(ExQ) * dsigma1v
    dHy0_dsig[1, :] = dHy0_dsig[2, :]
    dHy0_dsig[end, :] = dHy0_dsig[end-1, :]


    # Second interpolate fields to receiver locations (using linear interpolation)
    Exr = linRxMap' * Ex01[:,1]
    Hyr = linRxMap' * Hy0
    Hzr = linRxMap2' * (Bz0 ./ mu)

    dExr = linRxMap' * dEx0
    dHyr = linRxMap' * dHy0
    dHzr = linRxMap2' * sdiag(1 ./ mu) * dBz0

    dHyr_dsig = linRxMap' * dHy0_dsig


    #------- 2nd, computes the derivative of MT impedance or apparent resistivity
    # and phase or tipper with respect to EM fields.                    -------#
    Z = Exr ./ Hyr
    T = Hzr ./ Hyr

    dZ = sdiag(1 ./ Hyr) * dExr - sdiag(Exr ./ (Hyr .^2)) * dHyr
    dZ_dsig = -sdiag(Exr./(Hyr.^2)) * dHyr_dsig

    dT = sdiag(1 ./ Hyr) * dHzr - sdiag(Hzr ./ (Hyr .^2)) * dHyr
    dT_dsig = -sdiag(Hzr./(Hyr.^2)) * dHyr_dsig

    omu = omega*MU0

    nRx = length(Exr)

    if occursin("Impedance", dataType)
        # L = spzeros(ComplexF64, nRx, nNode)
        # Q = spzeros(ComplexF64, nRx, nCell)

        L = dZ          # (nRx, nNode)
        Q = dZ_dsig     # (nRx, nCell)

        if occursin("Tipper", dataType)
            L = vcat(L, dT)           # (2*nRx, nNode)
            Q = vcat(Q, dT_dsig)      # (2*nRx, nCell)
        end

    elseif occursin("Rho_Phs", dataType)
        # L = Array{ComplexF64}(nRx, nNode, 2)
        # Q = Array{Float64}(nRx, nCell, 2)

        # L = spzeros(ComplexF64, 2*nRx, nNode)
        # Q = spzeros(Float64, 2*nRx, nCell)

        appRho    = abs.(Z).^2 / omu
        # dAppRho   = 2 / omu * (diag(real(Z))*real(dZ) + diag(imag(Z))*imag(dZ))
	    dAppRho   = 2 / omu * sdiag(conj(Z)) * dZ
        dlog10Rho = sdiag( 1 ./ (log(10)*appRho) ) * dAppRho
        # dPhase    = diag(1 ./ abs(Z).^2) * (diag(real(Z))*imag(dZ) - diag(imag(Z))*real(dZ))
	    dPhase    = sdiag(1 ./ abs.(Z).^2) * (-1im) * sdiag(conj(Z)) * dZ
        dPhase    = dPhase * 180/pi

        dAppRho_dsig   = 2 / omu * (sdiag(real(Z))*real(dZ_dsig) + sdiag(imag(Z))*imag(dZ_dsig))
        dlog10Rho_dsig = sdiag(1 ./ (log(10)*appRho)) * dAppRho_dsig
        dPhase_dsig = sdiag(1 ./ abs.(Z).^2) * ( sdiag(real(Z))*imag(dZ_dsig) - sdiag(imag(Z))*real(dZ_dsig) )
        dPhase_dsig = dPhase_dsig * 180/pi

        L = vcat(dAppRho, dPhase)
        Q = vcat(dAppRho_dsig, dPhase_dsig)

        for j=1:length(dataComp)
            if occursin("log10Rho", dataComp[j])
                L[1:nRx, :] = dlog10Rho
                Q[1:nRx, :] = dlog10Rho_dsig
                break
            end
        end


        if occursin("Tipper", dataType)
            #dRealT = sdiag( real(T)./T ) * dT
            #dImagT = sdiag( imag(T)./T ) * dT

            #L = vcat(L, dRealT, dImagT)
            L = vcat(L, dT)
            Q = vcat(Q, real(dT_dsig), imag(dT_dsig))
        end


    end

    return L, Q

end


#-------------------------------------------------------------------------------
"""
`getDataFuncSensTM` computes partial derivative of MT data of a single frequency
  of TM mode with respect to solution fields (L) and model conductivities (Q).
  It is closely related to `compFieldsAtRxTM`.

Input:
    omega       :: Float64     - a single angular frequency value.
    rxSensInfo  :: rxSensInfo  - receiver locations.
    Hx01        :: Array       - the two row grid node Hx.
    dataType    :: String
    dataComp    :: Array{String}

Output:
    L :: Array
    Q :: Array

"""
function getDataFuncSensTM(omega::Float64, rxSensInfo::PreRxSens,
    Hx01::Array{ComplexF64}, dataType::String, dataComp::Array{String})

    #------- 1st, compute the partial derivative of EM fields at sites -------#

    # extract things
    dHx0      = rxSensInfo.dFn0
    dHx1      = rxSensInfo.dFn1
    sigma1    = rxSensInfo.sigma1
    dsigma1   = rxSensInfo.dsigma1
    yLen      = rxSensInfo.yLen
    zLen1     = rxSensInfo.zLen1
    linRxMap  = rxSensInfo.linRxMap

    ny    = length(yLen)
    nNode = size(dHx0, 2)
    nCell = size(dsigma1, 2)

    MU0 = 4 * pi * 1e-7
    mu = MU0 * ones(ny)


    # First compute fields at the receiver layer (earth surface or seafloor).
    Jz0 = -(ddx(ny) * Hx01[:,1]) ./yLen
    Jz1 = -(ddx(ny) * Hx01[:,2]) ./yLen
    dtmp = -sdiag(1 ./ yLen) * ddx(ny)
    dJz0 = dtmp * dHx0
    dJz1 = dtmp * dHx1

    # quarter Ez (1/4), with length ny
    EzQ = (0.75 * Jz0 + 0.25 * Jz1) ./ sigma1
    dEzQ = sdiag(1 ./ sigma1) * (0.75*dJz0 + 0.25*dJz1)
    dEzQ_dsig = sdiag(0.75*Jz0 + 0.25*Jz1) * sdiag(-1 ./ (sigma1.^2)) * dsigma1

    # half Ey (1/2), with length ny-1
    JyH   = (Hx01[2:end-1,2] - Hx01[2:end-1,1]) / zLen1
    rho1v = (avnc(ny-1)*((1 ./ sigma1).*yLen)) ./ (avnc(ny-1)*yLen)
    dJyH = (dHx1[2:end-1,:] - dHx0[2:end-1,:]) / zLen1
    EyH   = JyH .* rho1v
    dEyH = sdiag(rho1v) * dJyH
    drho1v = sdiag(1 ./  (avnc(ny-1)*yLen)) * avnc(ny-1) * sdiag(yLen) *
             sdiag(-1 ./ (sigma1.^2)) * dsigma1

    dEyH_dsig = sdiag(JyH) * drho1v

    # quarter Hx (1/4), with length ny-1
    HxQ = 0.75 * Hx01[2:end-1,1] + 0.25 * Hx01[2:end-1,2]
    dHxQ = 0.75 * dHx0[2:end-1,:] + 0.25 * dHx1[2:end-1,:]

    # average permeability at vertical edge
    # muv = (avnc(ny-1)*(mu.*yLen)) ./ (avnc(ny-1)*yLen)

    # dEz/dy
    dEzQ_dy = (ddx(ny-1)*EzQ) ./ (avnc(ny-1)*yLen)
    dtmp = sdiag(1 ./ (avnc(ny-1)*yLen)) * ddx(ny-1)
    ddEzQ = dtmp * dEzQ
    ddEzQ_dsig = dtmp * dEzQ_dsig

    # Faraday's law: -dEz/dy + dEy/dz = 1im*omega*muv.*HxQ,
    # where dEy/dz = (EyH-Ey0)/(0.5*zLen1).
    Ey0 = zeros(ComplexF64, ny+1)
    Ey0[2:end-1] = EyH - (dEzQ_dy + 1im*omega*MU0*HxQ)*(0.5*zLen1)
    Ey0[1]   = Ey0[2]
    Ey0[end] = Ey0[end-1]
    dEy0 = spzeros(ComplexF64, ny+1, nNode)
    dEy0[2:end-1, :] = dEyH - (ddEzQ + 1im*omega*MU0*dHxQ)*(0.5*zLen1)
    dEy0[1, :] = dEy0[2, :]
    dEy0[end, :]= dEy0[end-1, :]

    dEy0_dsig = spzeros(ComplexF64, ny+1, nCell)
    dEy0_dsig[2:end-1, :] = dEyH_dsig - ddEzQ_dsig*(0.5*zLen1)
    dEy0_dsig[1, :] = dEy0_dsig[2, :]
    dEy0_dsig[end, :] = dEy0_dsig[end-1, :]


    # Second interpolate fields to receiver locations (using linear interpolation)
    Hxr = linRxMap' * Hx01[:,1]
    Eyr = linRxMap' * Ey0

    dHxr = linRxMap' * dHx0
    dEyr = linRxMap' * dEy0

    dEyr_dsig = linRxMap' * dEy0_dsig


    #------- 2nd, computes the derivative of MT impedance or apparent resistivity
    # and phase with respect to EM fields.                             -------#
    Z = Eyr ./ Hxr

    dZ = sdiag(1 ./Hxr) * dEyr - sdiag(Eyr./(Hxr.^2)) * dHxr
    dZ_dsig = sdiag(1 ./ Hxr) * dEyr_dsig

    omu = omega*MU0

    nRx = length(Eyr)

    if occursin("Impedance", dataType)
        L = spzeros(ComplexF64, nRx, nNode)
        Q = spzeros(ComplexF64, nRx, nCell)

        L = dZ
        Q = dZ_dsig

    elseif occursin("Rho_Phs", dataType)
        # L = Array{ComplexF64}(nRx, nNode, 2)
        # Q = Array{Float64}(nRx, nCell, 2)

        L = spzeros(ComplexF64, 2*nRx, nNode)
        Q = spzeros(Float64, 2*nRx, nCell)

        appRho    = abs.(Z).^2 / omu
        # dAppRho   = 2 / omu * (diag(real(Z))*real(dZ) + diag(imag(Z))*imag(dZ))
        dAppRho   = 2 / omu * sdiag(conj(Z)) * dZ
        dlog10Rho = sdiag( 1 ./ (log(10)*appRho) ) * dAppRho
        # dPhase    = diag(1 ./ abs(Z).^2) * (diag(real(Z))*imag(dZ) - diag(imag(Z))*real(dZ))
        dPhase    = sdiag(1 ./ abs.(Z).^2) * (-1im) * sdiag(conj(Z)) * dZ
        dPhase    = dPhase * 180/pi

        dAppRho_dsig   = 2 / omu * (sdiag(real(Z))*real(dZ_dsig) + sdiag(imag(Z))*imag(dZ_dsig))
        dlog10Rho_dsig = sdiag(1 ./ (log(10)*appRho)) * dAppRho_dsig
        dPhase_dsig = sdiag(1 ./ abs.(Z).^2) * ( sdiag(real(Z))*imag(dZ_dsig) - sdiag(imag(Z))*real(dZ_dsig) )
        dPhase_dsig = dPhase_dsig * 180/pi

        # L[:, :, 1] = dAppRho
        # L[:, :, 2] = dPhase
        # Q[:, :, 1] = dAppRho_dsig
        # Q[:, :, 2] = dPhase_dsig

        L[1:nRx, :] = dAppRho
        L[nRx+1:2*nRx, :] = dPhase

        Q[1:nRx, :] = dAppRho_dsig
        Q[nRx+1:2*nRx, :] = dPhase_dsig

        for j=1:length(dataComp)
            if occursin("log10Rho", dataComp[j])
                L[1:nRx, :] = dlog10Rho
                Q[1:nRx, :] = dlog10Rho_dsig
                break
            end
        end

    end


    return L, Q

end
