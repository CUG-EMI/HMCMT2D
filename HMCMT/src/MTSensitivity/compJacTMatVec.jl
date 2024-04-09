export compJacTMatVec
#-------------------------------------------------------------------------------
"""
    `compJacTMatVec`
Computes Jacobian transpose matrix-vector product for 2D MT problem.

"""
function compJacTMatVec(exTE::Array{T}, hxTM::Array{T}, datVec::Vector,
                      mt2dMesh::TensorMesh2D,
                      mtData::MTData,
                      activeCell::SparseMatrixCSC=spzeros(0,0),
                      AinvTE::Array=zeros(0),
                      AinvTM::Array=zeros(0);
                      linSolver::String=""
                      ) where {T<:Complex}

    MU0 = 4 * pi * 1e-7

    # extract field values from input composite type
    yLen     = mt2dMesh.yLen
    zLen     = mt2dMesh.zLen
    origin   = mt2dMesh.origin
    sigma    = mt2dMesh.sigma
    ny, nz   = mt2dMesh.gridSize

    freqs    = mtData.freqs
    rxLoc    = mtData.rxLoc
    dataType = mtData.dataType
    dataComp = mtData.dataComp
    rxID     = mtData.rxID
    freqID   = mtData.freqID
    dtID     = mtData.dtID
    compTE   = mtData.compTE
    compTM   = mtData.compTM

    yNode = [0; cumsum(yLen)] .- origin[1]
    zNode = [0; cumsum(zLen)] .- origin[2]

    nFreq = length(freqs)
    nDC   = length(dataComp)
    nRx   = size(rxLoc, 1)

    nCell = ny*nz
    nNode = (ny+1)*(nz+1)

    isempty(activeCell) && ( activeCell = spunit(nCell) )

    nAC = size(activeCell, 2)

    mu = MU0 * ones(nCell)

    !mt2dMesh.setup && setupTensorMesh2D!(mt2dMesh)
    F     = mt2dMesh.Face
    Grad  = mt2dMesh.Grad
    AveCN = mt2dMesh.AveCN
    AveCF = mt2dMesh.AveCF

    if isempty(linSolver)
        lsFlag = 1
    elseif uppercase(linSolver) == "MUMPS"
        lsFlag = 2
    elseif uppercase(linSolver) == "PARDISO"
        lsFlag = 3
    else
        # iterative solvers, not yet developed.
    end


    # set the index of inner and outter/boundary part of coefficient matrix
    (ii, io) = getBoundaryIndex(ny, nz)

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
    JTv = zeros(ComplexF64, nAC)
    QTv = zeros(ComplexF64, nAC)

    # loop over frequencies to ...
    for iFreq=1:nFreq
        freq = freqs[iFreq]
        omega = 2 * pi * freq

        indF  = findall(freqID .== iFreq)

        # check if current frequency exists
        isempty(indF) && continue

        subRxID = rxID[indF]
        subDcID = dtID[indF]

        # sub-data vector for this frequency
        datTmp = conj(datVec[indF])
        #datTmp = copy(datVec[indF])

        # check which mode is present within this frequency.
        calTE = false
        calTM = false
        for j=1:length(subDcID)
            if occursin("XY", dataComp[subDcID[j]])
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
            # println("J^T*v: Solving pseudo forward problem for TE mode for Freq No. $iFreq/$nFreq ...")
            Ex01 = Array{ComplexF64}(undef, ny+1, 2)
            Ex01[:, 1]  = exTE[id0, iFreq]
            Ex01[:, 2]  = exTE[id1, iFreq]

            L, Q = getDataFuncSensTE(omega, rxSensInfo, Ex01, dataType, dataComp)

            if occursin("Rho_Phs", dataType)
                idd1 = findall(subDcID .== iRhoXY)
                idd2 = findall(subDcID .== iPhsXY)
                idr1 = subRxID[idd1]
                idr2 = subRxID[idd2] .+ nRx

                sVec = transpose(L[idr1, :]) * datTmp[idd1] + transpose(L[idr2, :]) * datTmp[idd2]

                QTv = QTv + activeCell' * ( transpose(Q[idr1, :]) * datTmp[idd1] +
                                            transpose(Q[idr2, :]) * datTmp[idd2] )

                # free memory
                L = 0
                Q = 0

            elseif occursin("Impedance", dataType)
                idd1 = findall(subDcID .== iZXY)
                idr1 = subRxID[idd1]

                sVec = transpose(L[idr1, :]) * datTmp[idd1]
                QTv = QTv + activeCell' * ( transpose(Q[idr1, :]) * datTmp[idd1] )

                # free memory
                L = 0
                Q = 0
            end

            AiiTE = rAiiTE + 1im * omega * iAiiTE
            AioTE = rAioTE + 1im * omega * iAioTE

            # solve the pseudo forward problem
            if lsFlag == 1
                eVal = AinvTE[iFreq] \ sVec[ii]

            elseif lsFlag == 2
                eVal = applyMUMPS(AinvTE[iFreq], sVec[ii])

            elseif lsFlag == 3
                eVal = pardiSolver(AiiTE, sVec[ii])

            end

            eVal_io = sVec[io]

            # The following is the new memory-efficient way to compute JTv,
            # neither P nor B is explicitly formed.
            PTv = -1im * omega * transpose(dMsigCN) * sdiag(exTE[ii, iFreq]) * eVal

            dBC, = getBCderivTE(freq, yLen, zLen, sigma)                # (nEio, nCell)
            dBC = dBC * activeCell

            BTvii = transpose(-AioTE) * eVal
            BTvii = transpose(dBC) * BTvii
            BTvio = transpose(dBC) * eVal_io

            JTv = JTv + PTv + BTvii + BTvio

            # @printf("  elapsed time: %8.4f %s\n", t, "seconds.")

        end  # calTE


        # for TM mode
        if calTM
            # println("J^T*v: Solving pseudo forward problem for TM mode for Freq No. $iFreq/$nFreq ...")
            Hx01 = Array{ComplexF64}(undef, ny+1, 2)
            Hx01[:, 1]  = hxTM[id0, iFreq]
            Hx01[:, 2]  = hxTM[id1, iFreq]

            L, Q = getDataFuncSensTM(omega, rxSensInfo, Hx01, dataType, dataComp)

            if occursin("Rho_Phs", dataType)
                idd1 = findall(subDcID .== iRhoYX)
                idd2 = findall(subDcID .== iPhsYX)
                idr1 = subRxID[idd1]
                idr2 = subRxID[idd2] .+ nRx

                sVec = transpose(L[idr1, :]) * datTmp[idd1] + transpose(L[idr2, :]) * datTmp[idd2]

                QTv = QTv + activeCell' * ( transpose(Q[idr1, :]) * datTmp[idd1] +
                                            transpose(Q[idr2, :]) * datTmp[idd2] )

                # free memory
                L = 0
                Q = 0

            elseif occursin("Impedance", dataType)
                idd1 = findall(subDcID .== iZYX)
                idr1 = subRxID[idd1]

                sVec = transpose(L[idr1, :]) * datTmp[idd1]
                QTv = QTv + activeCell' * ( transpose(Q[idr1, :]) * datTmp[idd1] )

                # free memory
                L = 0
                Q = 0
            end

            AiiTM = rAiiTM + 1im * omega * iAiiTM
            AioTM = rAioTM + 1im * omega * iAioTM

            # solve the pseudo forward problem
            if lsFlag == 1
                eVal = AinvTM[iFreq] \ sVec[ii]

            elseif lsFlag == 2
                eVal = applyMUMPS(AinvTM[iFreq], sVec[ii])

            elseif lsFlag == 3
                eVal = pardiSolver(AiiTM, sVec[ii])

            end

            eVal_io = sVec[io]

            # The following is the new memory-efficient way to compute JTv,
            # neither P nor B is explicitly formed.
            PTv = - Gradii * eVal
            PTv = transpose(dMsigF) * sdiag(Gradii*hxTM[ii, iFreq]) * PTv

            dBC, bc = getBCderivTM(freq, yLen, zLen, sigma)                  # (nEb, nCell)
            dBC = dBC * activeCell

            BTvii1 = transpose(-AioTM) * eVal
            BTvii1 = transpose(dBC) * BTvii1
            BTvii2 = - Gradii * eVal
            BTvii2 = transpose(dMsigF) * sdiag(Gradio*bc) * BTvii2
            BTvio  = transpose(dBC) * eVal_io

            JTv = JTv + PTv + BTvii1 + BTvii2 + BTvio

        end  # calTM


    end  # nFreq

    JTv = JTv + QTv

    return real(JTv)

end
