export compJacMat
#-------------------------------------------------------------------------------
"""
Computes full Jacobian matrix for 2D MT problem.

"""
function compJacMat(exTE::Array{T}, hxTM::Array{T}, mt2dMesh::TensorMesh2D,
                  datInfo::MTData, activeCell::SparseMatrixCSC=spzeros(0,0),
                  AinvTE::Array=zeros(0), AinvTM::Array=zeros(0); linSolver::String="") where {T<:Complex}

    MU0 = 4 * pi * 1e-7

    # extract field values from input composite type
    yLen     = mt2dMesh.yLen
    zLen     = mt2dMesh.zLen
    origin   = mt2dMesh.origin
    sigma    = mt2dMesh.sigma
    ny, nz   = mt2dMesh.gridSize

    freqs    = datInfo.freqs
    rxLoc    = datInfo.rxLoc
    dataType = datInfo.dataType
    dataComp = datInfo.dataComp
    rxID     = datInfo.rxID
    freqID   = datInfo.freqID
    dcID     = datInfo.dtID
    compTE   = datInfo.compTE
    compTM   = datInfo.compTM

    yNode = [0; cumsum(yLen)] .- origin[1]
    zNode = [0; cumsum(zLen)] .- origin[2]

    nFreq = length(freqs)
    nDC   = length(dataComp)
    nRx   = size(rxLoc, 1)

    nCell = ny*nz
    nNode = (ny+1)*(nz+1)

    isempty(activeCell) && ( activeCell = speye(nCell) )

    nAC = size(activeCell, 2)

    mu = MU0 * ones(nCell)

    !mt2dMesh.setup && setupTensorMesh2D!(mt2dMesh)
    F     = mt2dMesh.Face
    Grad  = mt2dMesh.Grad
    AveCN = mt2dMesh.AveCN
    AveCF = mt2dMesh.AveCF


    if isempty(AinvTE) || isempty(AinvTM)  # system matrix decomposition factors are not provided.
        if isempty(linSolver)
            lsFlag = 1
        elseif uppercase(linSolver) == "MUMPS"
            lsFlag = 2
        elseif uppercase(linSolver) == "PARDISO"
            lsFlag = 3
        else
            # iterative solvers, not yet developed.
        end

    else
        lsFlag = 0

    end


    # set the index of inner and outter/boundary part of coefficient matrix
    ii, io = getBoundaryIndex(ny, nz)

    # Prepare things for TE mode.
    if compTE
        # set up TE coefficient matrix
        MsigCN  = sdiag(AveCN * (F * sigma))
        MmuF    = sdiag(AveCF * (F *(1 ./ mu)))
        dGradTE = Grad' * MmuF * Grad  # double Grad
        rAiiTE  = dGradTE[ii, ii]
        rAioTE  = dGradTE[ii, io]
        iAiiTE  = MsigCN[ii, ii]
        iAioTE  = MsigCN[ii, io]      # is a zero matrix because MsigCN is diagonal
        # AveCNii = AveCN[ii, :]
        dMsigCN = AveCN[ii, :] * F * activeCell
    end


    # Prepare things for TM mode.
    if compTM
        # set up TM coefficient matrix
        MmuCN   = sdiag(AveCN * (F * mu))
        MsigF   = sdiag(AveCF * (F *(1 ./ sigma)))
        dGradTM = Grad' * MsigF * Grad
        rAiiTM  = dGradTM[ii, ii]
        rAioTM  = dGradTM[ii, io]
        iAiiTM  = MmuCN[ii, ii]
        iAioTM  = MmuCN[ii, io]      # is a zero matrix because MmuCN is diagonal
        Gradii  = Grad[:, ii]
        Gradio  = Grad[:, io]
        dMsigF = AveCF * F * sdiag(-1 ./ sigma.^2) * activeCell
    end


    # set up the data type dictionary
    if occursin("Rho_Phs", dataType)
        iRhoXY=0; iPhsXY=0; iRhoYX=0; iPhsYX=0; iRealTZY=0; iImagTZY=0;
        for j=1:nDC
            if occursin("RhoXY", dataComp[j])
                iRhoXY = j
            elseif dataComp[j] == "PhsXY"
                iPhsXY = j
            elseif occursin("RhoYX", dataComp[j])
                iRhoYX = j
            elseif dataComp[j] == "PhsYX"
                iPhsYX = j
            elseif dataComp[j] == "RealTZY"
                iRealTZY = j
            elseif dataComp[j] == "ImagTZY"
                iImagTZY = j
            end
        end

    elseif occursin("Impedance", dataType)
        iZXY=0;  iZYX=0; iTZY=0;
        for j=1:nDC
            if dataComp[j] == "ZXY"
                iZXY = j
            elseif dataComp[j] == "ZYX"
                iZYX = j
            elseif dataComp[j] == "TZY"
                iTZY = j
            end
        end

    end

    # pre-sets things needed to perform the interpolation. Independent of freqs.
    rxSensInfo = preSetRxFieldSens(rxLoc, yNode, zNode, sigma)
    zid = rxSensInfo.zid
    id0 = (zid-1)*(ny+1)+1:zid*(ny+1)
    id1 = zid*(ny+1)+1:(zid+1)*(ny+1)

    # pre-allocate
    # T1 = Union{Float64, ComplexF64}: doesn't work!
    if occursin("Rho_Phs", dataType)
        T1 = Float64
    elseif occursin("Impedance", dataType)
        T1 = ComplexF64
    end
    JMat = Array{T1}(undef, 0, nAC)


    # loop over frequencies to ...
    for iFreq=1:nFreq
        freq = freqs[iFreq]
        omega = 2 * pi * freq

        indF  = findall(freqID .== iFreq)

        # check if current frequency exists
        isempty(indF) && continue

        subRxID = rxID[indF]
        subDcID = dcID[indF]

        # the J matrix for this frequency
        nd  = length(subRxID)
        Jtmp = Array{T1}(undef, nd, nAC)

        # check which mode is present within this frequency.
        calTE = false
        calTM = false
        for j=1:length(subDcID)
            if occursin("XY", dataComp[subDcID[j]]) ||
               occursin("TZY", dataComp[subDcID[j]])
                calTE = true
                break
            end
        end
        for j=1:length(subDcID)
            if occursin("YX", dataComp[subDcID[j]])
                calTM = true
                break
            end
        end


        # for TE mode
        if calTE
            println("J: Solving pseudo forward problem for TE mode for Freq No. $iFreq/$nFreq ...")

            AiiTE = rAiiTE + 1im * omega * iAiiTE
            AioTE = rAioTE + 1im * omega * iAioTE

            dExTE = zeros(ComplexF64, nNode, nAC)

            # -d(Aii*e)/dsigma
            # P = -1im * omega * sdiag(exTE[ii, iFreq]) * dMsigCN     # (nEii, nAC)

            # B: drhs/dsigma
            dBC, = getBCderivTE(freq, yLen, zLen, sigma)              # (nEio, nCell)
		    dBC = dBC * activeCell
            # B = -AioTE * dBC                                          # (nEii, nAC)

            # P+B
            PplusB = -1im * omega * sdiag(exTE[ii, iFreq]) * dMsigCN - AioTE * dBC


            # solve the pseudo forward problem
            if lsFlag == 0
                @time dExTE[ii, :]  = applyMUMPS(AinvTE[iFreq], PplusB)

            elseif lsFlag == 1
                @time dExTE[ii, :] = AiiTE \ (PplusB)

            elseif lsFlag == 2
                @time dExTE[ii, :] = mumpsSolver(AiiTE, PplusB)

            elseif lsFlag == 3
                @time dExTE[ii, :] = pardiSolver(AiiTE, PplusB)

            end

            PplusB = 0.0   # free memory

            dExTE[io, :] = dBC

            Ex01 = Array{ComplexF64}(undef, ny+1, 2)
            Ex01[:, 1]  = exTE[id0, iFreq]
            Ex01[:, 2]  = exTE[id1, iFreq]

            L, Q = getDataFuncSensTE(omega, rxSensInfo, Ex01, dataType, dataComp)

            if dataType == "Rho_Phs"
                dTE = zeros(nRx, nAC, 2)
                dTE[:, :, 1] = real( L[1:nRx, :] * dExTE ) + Q[1:nRx, :] * activeCell                 # dAppRho/dsigma
                dTE[:, :, 2] = real( L[nRx+1:2*nRx, :] * dExTE ) + Q[nRx+1:2*nRx, :] * activeCell     # dPhase/dsigma

            elseif dataType == "Rho_Phs_Tipper"
                dTE = zeros(nRx, nAC, 4)
                dTE[:, :, 1] = real( L[1:nRx, :] * dExTE ) + Q[1:nRx, :] * activeCell                     # dAppRho/dsigma
                dTE[:, :, 2] = real( L[nRx+1:2*nRx, :] * dExTE ) + Q[nRx+1:2*nRx, :] * activeCell         # dPhase/dsigma
                dTE[:, :, 3] = real( L[2*nRx+1:3*nRx, :] * dExTE ) + Q[2*nRx+1:3*nRx, :] * activeCell     # dRealTZY/dsigma
                #dTE[:, :, 4] = real( L[3*nRx+1:4*nRx, :] * dExTE ) + Q[3*nRx+1:4*nRx, :] * activeCell     # dImagTZY/dsigma
                dTE[:, :, 4] = imag( L[2*nRx+1:3*nRx, :] * dExTE ) + Q[3*nRx+1:4*nRx, :] * activeCell     # dImagTZY/dsigma

            elseif dataType == "Impedance"
                dTE = L * dExTE + Q * activeCell    # dZ/dsigma

            elseif dataType == "Impedance_Tipper"
                dTE = zeros(ComplexF64, nRx, nAC, 2)
                dTE[:, :, 1] = L[1:nRx, :] * dExTE + Q[1:nRx, :] * activeCell                # dZ/dsigma
                dTE[:, :, 2] = L[nRx+1:2*nRx, :] * dExTE + Q[nRx+1:2*nRx, :] * activeCell    # dT/dsigma

            end

            dExTE = 0.0  # free memory

        end  # calTE


        # for TM mode
        if calTM
            println("J: Solving pseudo forward problem for TM mode for Freq No. $iFreq/$nFreq ...")

            AiiTM = rAiiTM + 1im * omega * iAiiTM
            AioTM = rAioTM + 1im * omega * iAioTM

            dHxTM = zeros(ComplexF64, nNode, nAC)

            # -d(Aii*e)/dsigma
            # P = - Gradii' * sdiag(Gradii*hxTM[ii, iFreq]) * dMsigF      # (nEii, nAC)

            # B: drhs/dsigma, rhs = -Aio * bc
            dBC, bc = getBCderivTM(freq, yLen, zLen, sigma)               # (nEio, nCell)
		    dBC = dBC * activeCell
            # B = -AioTM * dBC - Gradii'* sdiag(Gradio*bc) * dMsigF       # (nEii, nAC)

            # P+B
            PplusB = - Gradii' * sdiag(Gradii * hxTM[ii, iFreq]) * dMsigF -
                       AioTM * dBC - Gradii' * sdiag(Gradio*bc) * dMsigF


            # solve the pseudo forward problem
            if lsFlag == 0
                @time dHxTM[ii, :]  = applyMUMPS(AinvTM[iFreq], PplusB)

            elseif lsFlag == 1
                @time dHxTM[ii, :] = AiiTM \ (PplusB)

            elseif lsFlag == 2
                @time dHxTM[ii, :] = mumpsSolver(AiiTM, PplusB)

            elseif lsFlag == 3
                @time dHxTM[ii, :] = pardiSolver(AiiTM, PplusB)

            end

            PplusB = 0.0   # free memory

            dHxTM[io, :] = dBC

            Hx01 = Array{ComplexF64}(undef, ny+1, 2)
            Hx01[:, 1]  = hxTM[id0, iFreq]
            Hx01[:, 2]  = hxTM[id1, iFreq]

            L, Q = getDataFuncSensTM(omega, rxSensInfo, Hx01, dataType, dataComp)

            if size(L, 1)>nRx
                dTM = zeros(nRx, nAC, 2)
                dTM[:, :, 1] = real( L[1:nRx, :] * dHxTM ) + Q[1:nRx, :] * activeCell                # dAppRho/dsigma
                dTM[:, :, 2] = real( L[nRx+1:2*nRx, :] * dHxTM ) + Q[nRx+1:2*nRx, :] * activeCell    # dPhase/dsigma
            else
                dTM = L * dHxTM + Q * activeCell    # dZ/dsigma
            end

            dHxTM = 0.0  # free memory

        end  # calTM

        # map the rows of dTE and dTM to the right places in JMat
        # this is a much faster way in comparison with the row-by-row way below

        if calTE
            if occursin("Rho_Phs", dataType)
                idd1 = findall(subDcID .== iRhoXY)
                idd2 = findall(subDcID .== iPhsXY)
                idr1 = subRxID[idd1]
                idr2 = subRxID[idd2]
                Jtmp[idd1, :] = dTE[idr1, :, 1]
                Jtmp[idd2, :] = dTE[idr2, :, 2]

                if occursin("Tipper", dataType)
                    idd3 = findall(subDcID .== iRealTZY)
                    idd4 = findall(subDcID .== iImagTZY)
                    idr3 = subRxID[idd3]
                    idr4 = subRxID[idd4]
                    Jtmp[idd3, :] = dTE[idr3, :, 3]
                    Jtmp[idd4, :] = dTE[idr4, :, 4]
                end

            elseif occursin("Impedance", dataType)
                idd1 = findall(subDcID .== iZXY)
                idr1 = subRxID[idd1]

                if occursin("Tipper", dataType)
                    idd2 = findall(subDcID .== iTZY)
                    idr2 = subRxID[idd2]
                    Jtmp[idd1, :] = dTE[idr1, :, 1]
                    Jtmp[idd2, :] = dTE[idr2, :, 2]
                else
                    Jtmp[idd1, :] = dTE[idr1, :]
                end

            end
        end


	    if calTM
            if occursin("Rho_Phs", dataType)
                idd1 = findall(subDcID .== iRhoYX)
                idd2 = findall(subDcID .== iPhsYX)
                idr1 = subRxID[idd1]
                idr2 = subRxID[idd2]
                Jtmp[idd1, :] = dTM[idr1, :, 1]
                Jtmp[idd2, :] = dTM[idr2, :, 2]

            elseif occursin("Impedance", dataType)
                idd1 = findall(subDcID .== iZYX)
                idr1 = subRxID[idd1]
                Jtmp[idd1, :] = dTM[idr1, :]
            end
        end

        JMat = vcat(JMat, Jtmp)

    end  # nFreq

    return JMat

end
