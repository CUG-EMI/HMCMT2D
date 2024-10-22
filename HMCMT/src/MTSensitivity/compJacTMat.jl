export compJacTMat
#-------------------------------------------------------------------------------
"""
    `compJTMat`

Computes full Jacobian transpose matrix for 2D MT problem.

"""
function compJacTMat(exTE::Array{T}, hxTM::Array{T}, mt2dMesh::TensorMesh2D,
                   datInfo::MTData, activeCell::SparseMatrixCSC=spzeros(0,0),
                   AinvTE::Array=zeros(0), AinvTM::Array=zeros(0);
                   linSolver::String="") where {T<:Complex}

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

    isempty(activeCell) && ( activeCell = spunit(nCell) )

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
        dGradTE = spzeros(0,0)   # free memory
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
        dGradTM = spzeros(0,0)     # free memory
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
    JtMat = Array{T1}(undef, nAC, 0)

    # loop over frequencies to ...
    for iFreq=1:nFreq
        freq = freqs[iFreq]
        omega = 2 * pi * freq

        indF  = findall(freqID .== iFreq)
        # check if current frequency exists
        isempty(indF) && continue
        subRxID = rxID[indF]
        subDcID = dcID[indF]

        # sub-JT matrix for this frequency
        nd = length(subRxID)
        JTtmp = Array{ComplexF64}(undef, nAC, nd)
        QTtmp = Array{ComplexF64}(undef, nAC, nd)

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
            println("J^T: Solving pseudo forward problem for TE mode for Freq No. $iFreq/$nFreq ...")

            Ex01 = Array{ComplexF64}(undef, ny+1, 2)
            Ex01[:, 1]  = exTE[id0, iFreq]
            Ex01[:, 2]  = exTE[id1, iFreq]

            (L, Q) = getDataFuncSensTE(omega, rxSensInfo, Ex01, dataType, dataComp)

            if occursin("Rho_Phs", dataType)
                idd1 = findall(subDcID .== iRhoXY)
                idd2 = findall(subDcID .== iPhsXY)
                idr1 = subRxID[idd1]
                idr2 = subRxID[idd2] + nRx

                idte = [idd1; idd2]

                Lt = spzeros(ComplexF64, nNode, length(idte))
                Lt = hcat(copy(transpose(L[idr1, :]) ), copy(transpose(L[idr2, :])) )

                QTtmp[:, idd1] = copy(transpose( Q[idr1, :] * activeCell ) )
                QTtmp[:, idd2] = copy(transpose( Q[idr2, :] * activeCell ) )
                # QTtmp[:, idd1] = At_mul_Bt(activeCell, Q[idr1, :])
                # QTtmp[:, idd2] = At_mul_Bt(activeCell, Q[idr2, :])

                if occursin("Tipper", dataType)
                    idd3 = findall(subDcID .== iRealTZY)
                    idd4 = findall(subDcID .== iImagTZY)
                    idr3 = subRxID[idd3] + 2*nRx
                    idr4 = subRxID[idd4] + 2*nRx

                    idte = [idd1; idd2; idd3; idd4]

                    Lt = hcat(copy(transpose(Lt, L[idr3, :])), copy(transpose(L[idr4, :])) )

                    idr4 = idr4 + nRx
                    QTtmp[:, idd3] = copy(transpose( Q[idr3, :] * activeCell ) )
                    QTtmp[:, idd4] = copy(transpose( Q[idr4, :] * activeCell ) )

                end

                # free memory
                L = 0
                Q = 0

            elseif occursin("Impedance", dataType)


                idd1 = findall(subDcID .== iZXY)
                idr1 = subRxID[idd1]

                idte = idd1
                Lt = spzeros(ComplexF64, nNode, length(idte))
                Lt = copy(transpose(L[idr1, :]) )

                # QTtmp[:, idd1] = ( Q[idr1, :] * activeCell ).'
                QTtmp[:, idd1] = transpose(activeCell) * transpose(Q[idr1, :])

                if occursin("Tipper", dataType)
                    idd2 = findall(subDcID .== iTZY)
                    idr2 = subRxID[idd2] + nRx

                    idte = [idd1; idd2]
                    Lt = hcat(Lt, copy(transpose(L[idr2, :])) )
                    QTtmp[:, idd2] = transpose(activeCell) * transpose(Q[idr2, :])
                end

                # free memory
                L = 0
                Q = 0
            end


            AiiTE = rAiiTE + 1im * omega * iAiiTE
            AioTE = rAioTE + 1im * omega * iAioTE

            # eMat = Array{ComplexF64}(size(Lt))

            # solve the pseudo forward problem
            if lsFlag == 0
                # eMat[ii, :] = applyMUMPS(AinvTE[iFreq], Lt[ii, :])
                sTmp = applyMUMPS(AinvTE[iFreq], Lt[ii, :])

            elseif lsFlag == 1
                sTmp = AiiTE \ full(Lt[ii, :])

            elseif lsFlag == 2
                sTmp = mumpsSolver(AiiTE, Lt[ii, :])

            elseif lsFlag == 3
                sTmp = pardiSolver(AiiTE, full(Lt[ii, :]))

            end

            sTmp_io = Lt[io, :]

            Lt = 0   # free memory

            # The following is the new memory-efficient way to compute JTtmp,
            # neither P nor B is explicitly formed.
            Pt = -1im * omega * transpose(dMsigCN) * (sdiag(exTE[ii, iFreq]) * sTmp)

            dBC, = getBCderivTE(freq, yLen, zLen, sigma)                # (nEio, nCell)
            dBC = dBC * activeCell

            Btii = transpose(-AioTE) * sTmp
            Btii = transpose(dBC) * Btii
            Btio = transpose(dBC) * sTmp_io

            # -1im * omega * dMsigCN' * sdiag(exTE[ii, iFreq]) * sTmp -
            # dBC.' * AioTE.' * sTmp + dBC.' * sTmp_io

            JTtmp[:, idte] = Pt + Btii + Btio

            # Particularly, for data type of "Rho_Phs_Tipper", the derivative of tipper
            # need to be split into real part and imaginary part.
            if dataType == "Rho_Phs_Tipper"
                JTtmp[:, idd3] = real.(JTtmp[:, idd3])
                JTtmp[:, idd4] = imag.(JTtmp[:, idd4])
            end

        end  # calTE

        # for TM mode
        if calTM
            println("J^T: Solving pseudo forward problem for TM mode for Freq No. $iFreq/$nFreq ...")

            Hx01 = Array{ComplexF64}(undef, ny+1, 2)
            Hx01[:, 1]  = hxTM[id0, iFreq]
            Hx01[:, 2]  = hxTM[id1, iFreq]

            L, Q = getDataFuncSensTM(omega, rxSensInfo, Hx01, dataType, dataComp)

            if occursin("Rho_Phs", dataType)
                idd1 = findall(subDcID .== iRhoYX)
                idd2 = findall(subDcID .== iPhsYX)
                idr1 = subRxID[idd1]
                idr2 = subRxID[idd2] + nRx

                idtm = [idd1; idd2]
                Lt = spzeros(ComplexF64, nNode, length(idtm))
                Lt = hcat(copy(transpose(L[idr1, :])), copy(transpose(L[idr2, :])) )

                QTtmp[:, idd1] = copy(transpose( Q[idr1, :] * activeCell ) )
                QTtmp[:, idd2] = copy(transpose( Q[idr2, :] * activeCell ) )

                # free memory
                L = 0
                Q = 0

            elseif occursin("Impedance", dataType)
                idd1 = findall(subDcID .== iZYX)
                idr1 = subRxID[idd1]

                idtm = idd1
                Lt = spzeros(ComplexF64, nNode, length(idtm))
                Lt = copy(transpose(L[idr1, :]) )

                QTtmp[:, idd1] = copy(transpose( Q[idr1, :] * activeCell ) )

                # free memory
                L = 0
                Q = 0
            end

            AiiTM = rAiiTM + 1im * omega * iAiiTM
            AioTM = rAioTM + 1im * omega * iAioTM


            # solve the pseudo forward problem
            if lsFlag == 0
                sTmp = applyMUMPS(AinvTM[iFreq], Lt[ii, :])

            elseif lsFlag == 1
                sTmp = AiiTM \ full(Lt[ii, :])

            elseif lsFlag == 2
                sTmp = mumpsSolver(AiiTM, Lt[ii, :])

            elseif lsFlag == 3
                sTmp = pardiSolver(AiiTM, full(Lt[ii, :]))

            end


            sTmp_io = Lt[io, :]
            Lt = 0

            # The following is the new memory-efficient way to compute JTtmp,
            # neither P nor B is explicitly formed.
            Pt = - Gradii * sTmp
            Pt = transpose(dMsigF) * (sdiag(Gradii*hxTM[ii, iFreq]) * Pt)

            dBC, bc = getBCderivTM(freq, yLen, zLen, sigma)                  # (nEio, nCell)
            dBC = dBC * activeCell

            Btii1 = transpose(-AioTM) * sTmp
            Btii1 = transpose(dBC) * Btii1
            Btii2 = - Gradii * sTmp
            Btii2 = transpose(dMsigF) * (sdiag(Gradio*bc) * Btii2)
            Btio  = transpose(dBC) * sTmp_io

            JTtmp[:, idtm] = Pt + Btii1 + Btii2 + Btio

        end  # calTM



        JTtmp = JTtmp + QTtmp
        JtMat = hcat(JtMat, JTtmp)

    end  # nFreq

    occursin("Rho_Phs", dataType) && ( JtMat = real(JtMat) )

    return JtMat

end
