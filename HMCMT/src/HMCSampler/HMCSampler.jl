#-------------------------------------------------------------------------------
# module `HMCSampler` defines routines to perform HMC sampling for 2D MT inversion.
# (c) Peng Ronghua, July, 2022
#
#-------------------------------------------------------------------------------
module HMCSampler

using SparseArrays, LinearAlgebra, Printf, Random
using Distributed, DistributedArrays
using Statistics, MUMPS
using HMCMT.HMCFileIO
using HMCMT.MTFwdSolver
using HMCMT.MTSensitivity
using HMCMT.HMCUtility
using HMCMT.HMCStruct

export copyHMCParam, updateHMCParam!
export runHMCSampler, proposeLeapfrog
export compDataGradient, getKineticGradient
export getHamiltonian, getKineticEnergy
export getMomentumVector, setMassMatrix
export checkParameterBound!, getBoundedParameter
export getPosteriorModel, outputPosterior
export outputHMCmodel, outputHMCSamples
export updateStartModel

# ENV["OMP_NUM_THREADS"] = 16
# ENV["MKL_NUM_THREADS"] = 16

#-------------------------------------------------------------------------------
"""
    `copyHMCParam(hmcParam)`

"""
function copyHMCParam(hmcParam::HMCParameter)

    #
    nparam   = hmcParam.nparam
    rhomodel = copy(hmcParam.rhomodel)
    momentum = copy(hmcParam.momentum)
    invM     = copy(hmcParam.invM)
    sqrtM    = copy(hmcParam.sqrtM)
    hmcParamNew = HMCParameter(nparam,rhomodel,momentum,invM,sqrtM)

    return hmcParamNew

end


#-------------------------------------------------------------------------------
"""
    `updateHMCParam!(hmcParam,rhomodel,momentum)`

"""
function updateHMCParam!(hmcParam::HMCParameter, rhomodel::Vector{T},
            momentum::Vector{T}) where{T<:Float64}

    #
    hmcParam.rhomodel = copy(rhomodel)
    hmcParam.momentum = copy(momentum)

    return hmcParam

end


#-------------------------------------------------------------------------------
"""
    `runHMCSampler(mtMesh,mtData,invParam,hmcprior)`

"""
function runHMCSampler(mtMesh::TensorMesh2D,
                       mtData::MTData,
                       invParam::InvDataModel,
                       hmcprior::HMCPrior)

    # initialize HMC parameter
    nparam   = length(invParam.strModel)
    ndata    = length(invParam.obsData)
    hmcParamCurrent  = initHMCParameter(nparam)
    if hmcprior.massType == "diagonal"
        scaling = 1.0 #/ (hmcprior.sigmastd * hmcprior.sigmastd)
        (invM, sqrtM) = setMassMatrix(nparam, scaling)
    else
        (invM, sqrtM) = setMassMatrix(invParam)
    end

    hmcParamCurrent.rhomodel = copy(invParam.strModel);
    hmcParamCurrent.invM     = copy(invM);
    hmcParamCurrent.sqrtM    = copy(sqrtM);
    hmcParamCurrent.momentum = getMomentumVector(nparam, hmcParamCurrent);
    hmcParamProposed = copyHMCParam(hmcParamCurrent);

    #
    startM  = 0; finishM = 0
    startK  = 0; finishK = 0
    startH  = 0; finishH = 0

    # set different starting model for parallel HMC sampling, added in 24 July, 2023
    sigma0 = unique(invParam.strModel)
    rho0   = 1.0 / exp(sigma0[1])
    rhomin = rho0 * 0.5
    rhomax = rho0 * 1.5
    rhoref = round(unirandDouble(rhomin, rhomax))
    println("Homogeneous starting model with a resistivity of $(rhoref) Ωm is used.")
    strModel = 1.0 / rhoref * ones(nparam)
    strModel = log.(strModel)
    invParam.strModel = copy(strModel)
    invParam.refModel = copy(strModel)
    #

    # compute Hamiltonian at current state
    mtMesh,_ = updateStartModel(mtMesh, invParam);
    (startD,startK,startH,startM,predData) = getHamiltonian(mtData,mtMesh,
                                              invParam,hmcprior,hmcParamCurrent);

    #
    iterNo   = 0
    nsamples = hmcprior.totalsamples
    hmcmodel = zeros(nparam, nsamples)
    hmcdata  = zeros(ComplexF64, ndata, nsamples+1)
    hmcstats = initHMCStatus(nsamples)
    hmcstats.hmstats[1,1] = startD
    hmcstats.hmstats[2,1] = startM
    hmcstats.hmstats[3,1] = startK
    hmcstats.hmstats[4,1] = startH
    hmcdata[:, 1] = copy(predData)

    # momentumstat = zeros(nparam, nsamples)
    # print info
    @printf("iterNo=%6d,dtMisfit=%8.3e,mNorm=%8.3e,KEnergy=%8.3e,HEnergy=%8.3e\n",iterNo,startD,startM,startK,startH)
    while (iterNo < nsamples)

        iterNo += 1
        # propose model and momentum parameters
        @time (propModel,propMomentum) = proposeLeapfrog(hmcParamCurrent,mtMesh,
                                                       mtData,invParam,hmcprior)
        hmcParamProposed = updateHMCParam!(hmcParamProposed,propModel,propMomentum)

        # compute Hamiltonian at proposed state
        (finishD,finishK,finishH,finishM,predData) = getHamiltonian(mtData,
                                      mtMesh,invParam,hmcprior,hmcParamProposed)

        # print info
        @printf("iterNo=%6d,old status: dtMisfit=%8.3e,mNorm=%8.3e,KEnergy=%8.3e,HEnergy=%8.3e\n",iterNo,startD,startM,startK,startH)
        @printf("iterNo=%6d,new status: dtMisfit=%8.3e,mNorm=%8.3e,KEnergy=%8.3e,HEnergy=%8.3e\n",iterNo,finishD,finishM,finishK,finishH)

        # check acceptance
        hdif = startH - finishH
        aratio = rand()
        if (hdif > 0) || (aratio < exp(hdif))
            hmcParamCurrent = updateHMCParam!(hmcParamCurrent,propModel,propMomentum)
            startD = finishD
            startM = finishM
            #
            hmcstats.nAccept += 1
            hmcstats.acceptstats[iterNo] = true
            tmp = exp(hdif)
            printstyled("Accepted with acceptance probability $(tmp).\n", color=:cyan)
            # save predicted data
            hmcdata[:, iterNo+1] = copy(predData)

        else
            hmcstats.nReject += 1
            tmp = exp(hdif)
            printstyled("Rejected with acceptance probability $(tmp).\n", color=:red)

            # save predicted data
            hmcdata[:, iterNo+1] = copy(hmcdata[:, iterNo])

        end

        # propse new momentum
        currMomentum = getMomentumVector(nparam, hmcParamCurrent)
        hmcParamCurrent.momentum = copy(currMomentum)
        startK = getKineticEnergy(currMomentum, hmcParamCurrent)
        startH = startD + startM + startK

        # record info
        hmcstats.hmstats[1,iterNo+1] = startD
        hmcstats.hmstats[2,iterNo+1] = startM
        hmcstats.hmstats[3,iterNo+1] = startK
        hmcstats.hmstats[4,iterNo+1] = startH

        # collect current model
        hmcmodel[:,iterNo] = copy(hmcParamCurrent.rhomodel)

        # update prior reference model, 24 July, 2023
        # prefModel = mean(hmcmodel[:,1:iterNo], dims=2)
        # invParam.refModel = copy(prefModel)

    end

    return hmcmodel, hmcstats, hmcdata

end


#-------------------------------------------------------------------------------
"""
    `proposeLeapfrog(hmcParamCurrent,mtMesh,mtData,invParam,hmcprior)`

get model and momentum proposals by leapfrog integrator.

"""
function proposeLeapfrog(hmcParamCurrent::HMCParameter,
                         mtMesh::TensorMesh2D,
                         mtData::MTData,
                         invParam::InvDataModel,
                         hmcprior::HMCPrior)

    #
    currModel    = hmcParamCurrent.rhomodel
    currMomentum = hmcParamCurrent.momentum
    invParam.strModel = copy(currModel)
    (predData,dataMisfit,dataGrad) = compDataGradient(mtMesh,mtData,invParam,hmcprior)
    hmcprior.nfevals += 1

    # prior model
    refModel  = invParam.refModel
    nparam    = length(invParam.refModel)
    Wm        = invParam.Wm
    modelGrad = Wm * (currModel - refModel) * hmcprior.regParam
    dataGrad  = dataGrad + modelGrad

    #
    dt = hmcprior.dt
    propMomentum = currMomentum - 0.5 * dt * dataGrad
    propModel    = copy(currModel)

    # perform numerical integrator using leapfrog scheme
    timestep = hmcprior.timestep
    intstep  = unirandInteger(timestep[1], timestep[2])
    maxStepSize = 3.0
    for k = 1:intstep

        gradK = getKineticGradient(propMomentum, hmcParamCurrent)
        dm    = dt*gradK
        # scale step for safe size
        dmMax = maximum(abs.(dm))
        if dmMax > maxStepSize
            dm = dm / dmMax * maxStepSize
        end
        propModel += dm

        # check if within predefined bounds
        (propModel,propMomentum) = checkParameterBound!(propModel,propMomentum,hmcprior)

        # update model and compute gradient at proposed model
        invParam.strModel = copy(propModel)
        (predData,dataMisfit,dataGrad) = compDataGradient(mtMesh,mtData,invParam,hmcprior)
        hmcprior.nfevals += 1

        # prior model
        modelGrad = Wm * (propModel - refModel) * hmcprior.regParam
        dataGrad  = dataGrad + modelGrad

        delta = dt * dataGrad
        if k < intstep
            propMomentum -= delta
        else
            propMomentum -= 0.5*delta
        end

    end

    return propModel, propMomentum

end


#-------------------------------------------------------------------------------
"""
    `compDataGradient(mtMesh,mtData,invParam,hmcprior)`

"""
function compDataGradient(mtMesh::TensorMesh2D, mtData::MTData,
                          invParam::InvDataModel, hmcprior::HMCPrior)

    # conductivity bounds
    sigmin = hmcprior.sigBounds[1]
    sigmax = hmcprior.sigBounds[2]
    solver = hmcprior.linearSolver

    # active cell
    activeCell = invParam.activeCell
    bgModel    = invParam.bgModel

    # convert log conductivity into linear conductivity
    strMod = invParam.strModel
    # (sigma, dsigma) = modelTransform(strMod, sigmin, sigmax)
    (sigma, dsigma) = modelTransform(strMod)
    sigma = activeCell * sigma + bgModel
    mtMesh.sigma = sigma
    (predData,fwdInfo) = MT2DFwdSolver(mtMesh,mtData,linearSolver=solver)

    # data residual res = Wd*(predData-obsData)
    obsData = invParam.obsData
    dataW   = invParam.dataW
    dataRes    = dataW * (predData - obsData)
    dataMisfit = getDataMisfit(dataRes)

    # data misfit gradient dphi_d(m)=Jᵀ * Wdᵀ * Wd * (F[m]-d\^{obs})
    dataRes = dataW' * dataRes
    dataGrad = compDataGradient(dataRes,fwdInfo,mtMesh,mtData,activeCell)
    dataGrad = dsigma' * dataGrad

    # destroy
    if isempty(solver)
        fwdInfo.AinvTE = zeros(0)
        fwdInfo.AinvTM = zeros(0)
    elseif lowercase(solver) == "mumps"
        nFreq = length(mtData.freqs)
        if mtData.compTE
            for j = 1:nFreq
                destroyMUMPS(fwdInfo.AinvTE[j])
            end
        end
        if mtData.compTM
            for j = 1:nFreq
                destroyMUMPS(fwdInfo.AinvTM[j])
            end
        end

    end

    return predData, dataMisfit, dataGrad


end


#-------------------------------------------------------------------------------
function compDataGradient(datVec::Vector, fwdInfo::MT2DFwdData,
                          mtMesh::TensorMesh2D, mtData::MTData,
                          activeCell::SparseMatrixCSC=spzeros(0,0))

#
exTE = fwdInfo.exTE
hxTM = fwdInfo.hxTM
linearSolver = fwdInfo.linearSolver
datGrad = compJacTMatVec(exTE,hxTM,datVec,mtMesh,mtData,activeCell,
      fwdInfo.AinvTE,fwdInfo.AinvTM,linSolver=linearSolver)

#
return datGrad

end


#-------------------------------------------------------------------------------
"""
    `getHamiltonian(mtData,mtMesh,invParam,hmcParam)`

calculate the energy of the Hamiltonian system.

"""
function getHamiltonian(mtData::MTData, mtMesh::TensorMesh2D,
                        invParam::InvDataModel, hmcprior::HMCPrior,
                        hmcParam::HMCParameter)

    # first compute predicted data for current model m
    solver = hmcprior.linearSolver
    (predData,fwdInfo) = MT2DFwdSolver(mtMesh, mtData, linearSolver=solver)
    # free memory
    if isempty(solver)
        fwdInfo.AinvTE = zeros(0)
        fwdInfo.AinvTM = zeros(0)
    elseif lowercase(solver) == "mumps"
        nFreq = length(mtData.freqs)
        if mtData.compTE
            for j = 1:nFreq
                destroyMUMPS(fwdInfo.AinvTE[j])
            end
        end
        if mtData.compTM
            for j = 1:nFreq
                destroyMUMPS(fwdInfo.AinvTM[j])
            end
        end

    end

    dataMisfit = compDataMisfit(predData, invParam)
    momentum   = hmcParam.momentum
    kp         = getKineticEnergy(momentum, hmcParam)

    #
    Wm     = invParam.Wm
    mprior = invParam.strModel - invParam.refModel
    mnorm  = 0.5 * mprior' * Wm * mprior * hmcprior.regParam

    hmp = dataMisfit  + kp + mnorm

    return dataMisfit, kp, hmp, mnorm, predData

end


#-------------------------------------------------------------------------------
"""
    `getKineticEnergy(momentum, hmcParam)`

compute the kinetic energy of the proposed model.

"""
function getKineticEnergy(momentum::Vector{T}, hmcParam::HMCParameter) where{T<:Float64}

    #
    tmp = hmcParam.invM * momentum
    kp  = 0.5 * dot(momentum, tmp)

    return kp

end


#-------------------------------------------------------------------------------
"""
    `getKineticGradient(momentum, hmcParam)`

compute the gradient of the kinetic energy.
"""
function getKineticGradient(momentum::Vector{T}, hmcParam::HMCParameter) where{T<:Float64}

    #
    gradK = hmcParam.invM * momentum

    return gradK

end

#-------------------------------------------------------------------------------
"""
    `getMomentumVector(nparam, massMatrix)`

propose momentum vector `p`, which are drawn randomly from a multivariate Gaussian
with covariance matrix `M`.

"""
function getMomentumVector(nparam::Int, hmcParam::HMCParameter)

    #
    maxVal = 2.5
    mp  = randn(nparam)
    ind = abs.(mp) .> maxVal
    mp[ind] = sign.(mp[ind]) * maxVal

    mp = hmcParam.sqrtM * mp

    return mp

end


#-------------------------------------------------------------------------------
"""
    `setMassMatrix(nparam, scaling)`

propose diagonal mass matrix `M`.

"""
function setMassMatrix(nparam::Int, scaling::Real)

    #
    massMatrix = scaling * ones(nparam)
    sqrtM = sqrt.(massMatrix)
    invM  = 1.0 ./ massMatrix
    invM  = Diagonal(invM)
    sqrtM = Diagonal(sqrtM)

    return invM, sqrtM

end


#-------------------------------------------------------------------------------
function setMassMatrix(invParam::InvDataModel)

    #
    massMatrix = Matrix(invParam.Wm)
    decomp = cholesky(massMatrix)
    sqrtM  = decomp.L
    Linv   = inv(decomp.L)
    invM   = Linv' * Linv

    return invM, sqrtM

end

#-------------------------------------------------------------------------------
"""
    `compDataMisfit(predData, invParam)`

compute data misfit.

"""
function compDataMisfit(predData::Array{ComplexF64}, invParam::InvDataModel)

    obsData = invParam.obsData
    dataW   = invParam.dataW
    dataRes    = dataW * (predData - obsData)
    dataMisfit = getDataMisfit(dataRes)

    return dataMisfit

end


#-------------------------------------------------------------------------------
"""
    `checkParameterBound(model,momentum,hmcprior)`

"""
function checkParameterBound!(model::Vector{T},momentum::Vector{T},
                              hmcprior::HMCPrior) where{T<:Float64}

    #
    sigmin = log(hmcprior.sigBounds[1])
    sigmax = log(hmcprior.sigBounds[2])

    #
    for k = 1:length(model)

        if (model[k] <= sigmax) & (model[k] >= sigmin)
            continue
        else
            outconstrained = true
            niter = 0
            while outconstrained
                niter += 1
                if model[k] < sigmin
                    model[k]     = 2.0*sigmin - model[k]
                    momentum[k] *= -1.0
                end
                #
                if model[k] > sigmax
                    model[k]     = 2.0*sigmax - model[k]
                    momentum[k] *= -1.0
                end
                #
                if (model[k] <= sigmax) & (model[k] >= sigmin)
                    outconstrained = false
                end
                #
                if niter >= 500
                    printstyled("constraints for $k is not fullfilled! The value is $(model[k]).\n",color=:red)
                end

            end # while
        end

    end
    #

    return model, momentum


end


#-------------------------------------------------------------------------------
"""
    `getBoundedParameter!(model, momentum, hmcprior)`

transforms the proposed model parameters into predefined model bounds.

"""
function getBoundedParameter!(model::Vector{T}, momentum::Vector{T},
                             hmcprior::HMCPrior) where{T<:Float64}

    #
    sigmin = log(hmcprior.sigBounds[1])
    sigmax = log(hmcprior.sigBounds[2])
    iterMax = 1000

    # transforms the candidate model to bounded constraints
    for k = 1:length(model)
        niter = 0
        while (model[k] > sigmax) | (model[k] < sigmin)
            if model[k] > sigmax
                model[k] = 2.0 * sigmax - model[k]
                momentum[k] *= -1.0
            elseif model[k] < sigmin
                model[k] = 2.0 * sigmin - model[k]
                momentum[k] *= -1.0
            end
            niter += 1
            if niter > iterMax
                printstyled("The conductivity value $(model[k]) for cell#$(k) out of the model bounds!",color=:red)
                error("Constrained HMC is failed!")
            end

        end #while

    end

    return model, momentum

end



#-------------------------------------------------------------------------------
function getPosteriorModel(hmcmodel::Array{Float64}, mtMesh::TensorMesh2D,
                           invParam::InvDataModel, hmcprior::HMCPrior)

    #
    burnin = hmcprior.burninsamples
    (nparam,nsamples) = size(hmcmodel)
    meanModel = zeros(nparam)
    stdModel  = zeros(nparam)

    #
    nEnsemble = 0
    for k = burnin+1:nsamples
        meanModel += hmcmodel[:, k]
        stdModel  += hmcmodel[:, k] .^ 2
        nEnsemble += 1
    end
    meanModel /= nEnsemble
    stdModel  /= nEnsemble

    # standard deviation
    std = stdModel - meanModel .^ 2
    std[std .< 0] .= eps(1.0)
    stdModel = sqrt.(std)

    # output posterior models
    activeCell = invParam.activeCell
    bgModel    = invParam.bgModel
    (sigma, dsigma) = modelTransform(meanModel)
    sigma = activeCell * sigma + bgModel
    mtMesh.sigma = sigma
    writeEMModel2D("meanModel.model", mtMesh)

    #
    sigma = activeCell * stdModel + bgModel
    mtMesh.sigma = sigma
    writeEMModel2D("stdModel.model", mtMesh)

end


#-------------------------------------------------------------------------------
function getPosteriorModel(hmcmodel::Array{Float64}, mtData::MTData,
                           mtMesh::TensorMesh2D, hmcprior::HMCPrior,
                           yBins::Vector{T}, zBins::Vector{T}) where {T<:Float64}

    #
    burnin = hmcprior.burninsamples
    (nparam,nsamples) = size(hmcmodel)

    sigmin = log10(hmcprior.sigBounds[1])
    sigmax = log10(hmcprior.sigBounds[2])
    npBins = 300
    (rhoBins,_) = mksampleArray(sigmin, sigmax, npBins)
    nyBins = length(yBins)
    nzBins = length(zBins)

    #
    rxLocy = emData.rxLoc[:, 1]
    nsite  = length(rxLocy)
    sitePosterior  = zeros(Int, nzBins, npBins, nsite)

    siteModel = zeros(nsite, nzBins)
    for k in burnin+1:nsamples
        siteModel = model2DInterp(rxLocy, zBins, mtMesh, hmcmodel[:, k])
        for isite = 1:nsite
            for iz = 1:nzBins
                idx = findLocation1D(siteModel[isite,iz], rhoBins)
                sitePosterior[iz, idx, isite] += 1
            end
        end
    end

    filename = "sitePPD.dat"
    outputPosterior(filename, rxLocy, zBins, rhoBins, sitePosterior)

end


#-------------------------------------------------------------------------------
function model2DInterp(mtMesh::TensorMesh2D,ycoord::Vector,zcoord::Vector,rho::Array)

    # cell information
    yNode = cumsum([0; mtMesh.yLen]) .- mtMesh.origin[1]
    zNode = cumsum([0; mtMesh.zLen]) .- mtMesh.origin[2]
    yCen = yNode[1:end-1] + diff(yNode) / 2
    zCen = zNode[1:end-1] + diff(zNode) / 2

    ny = length(mtMesh.yLen)
    nz = length(mtMesh.zLen) - length(mtMesh.airLayer)

    rhoModel = reshape(rho, ny, nz)

    # interpolate to the nearest node, not by linear interpolation
    ny = length(ycoord)
    nz = length(zcoord)
    interpModel = zeros(ny, nz)
    for k = 1:nz
        for j = 1:ny
            point = [ ycoord[j] zcoord[k] ]
            idx   = findLocation2D(point, yCen, zCen)
            interpModel[j, k] = rhoModel[idx[1], idx[2]]
        end
    end

    return interpModel

end


#-------------------------------------------------------------------------------
"""
    `outputPosterior(coordBins, rhoBins, ncellPosterior, valuePosterior)`

"""
function outputPosterior(filename::String, yBins::Vector{T}, zBins::Vector{T},
                      pBins::Vector{T}, valuePosterior::Array) where{T<:Float64}

    #
    ndim = size(valuePosterior)
    if ndim[3] != length(yBins) || ndim[2] != length(zBins) || ndim[1] == length(pBins)
        error("Dimension mismatch!")
    end

    fileID = open(filename, "w")
    @printf(fileID, "y coordinate: %5d\n", ndim[3])
    for k = 1:ndim[3]
        @printf(fileID, "%6g ", yBins[k])
    end
    @printf(fileID, "\n")
    @printf(fileID, "z coordinate: %5d\n", ndim[2])
    for k = 1:ndim[2]
        @printf(fileID, "%6g ", zBins[k])
    end
    @printf(fileID, "\n")
    @printf(fileID, "y coordinate: %5d\n", ndim[1])
    for k = 1:ndim[1]
        @printf(fileID, "%6g ", pBins[k])
    end
    @printf(fileID, "\n")
    #
    for k = 1:ndim[3]
        for j = 1:ndim[2]
            for i = 1:ndim[1]
                @printf(fileID, "%5d ", valuePosterior[i,j,k])
            end
            @printf(fileID, "\n")
        end
        @printf(fileID, "\n")
    end
    close(fileID)

end


#-------------------------------------------------------------------------------
function outputHMCmodel(hmcmodel::Array{Float64}, mtMesh::TensorMesh2D,
                        invParam::InvDataModel, start::Int=1, step::Int=10)

    #
    (nparam,nsamples) = size(hmcmodel)

    # output hmc statistical models
    activeCell = invParam.activeCell
    bgModel    = invParam.bgModel

    for k = start:step:nsamples
        (sigma, dsigma) = modelTransform(hmcmodel[:,k])
        sigma = activeCell * sigma + bgModel
        mtMesh.sigma = sigma
        writeEMModel2D("hmcmodel_iter$(k).model", mtMesh)
    end

end


#-------------------------------------------------------------------------------
"""
    `outputHMCSamples(hmcmodel, hmcstats)`

"""
function outputHMCSamples(hmcmodel::Array{Float64}, hmcstats::HMCStatus,
                          hmcdata::Array{ComplexF64}; ichain::Int=1, cputime::Float64=0.0)

    #
    (nparam,nsamples) = size(hmcmodel)

    fileID  = open("hmcsamples_id$(ichain).model", "w")
    for k = 1:nsamples
        for j = 1:nparam
            @printf(fileID, "%8.4e ", hmcmodel[j,k])
        end
        @printf(fileID, "\n")
    end
    close(fileID)

    # output predicted data
    fileID  = open("hmcsamples_id$(ichain).data", "w")
    for k = 1:nsamples+1
        for j = 1:size(hmcdata,1)
            @printf(fileID, "%12.4e %12.4e", real(hmcdata[j,k]),imag(hmcdata[j,k]))
        end
        @printf(fileID, "\n")
    end
    close(fileID)

    # output hmc statistics
    hmstats = hmcstats.hmstats
    fileID  = open("hmcstatistics_id$(ichain).log", "w")
    @printf(fileID, "Total elapsed time (s): %8.2f\n", cputime)
    @printf(fileID, "Totalsamples: %6d, nAccept: %6d, nReject: %6d\n", nsamples,
            hmcstats.nAccept, hmcstats.nReject)
    @printf(fileID, "Starting status: dtMisfit=%8.1f,mNorm=%8.1f,KEnergy=%8.1f,HEnergy=%8.1f\n",
            hmstats[1,1],hmstats[2,1],hmstats[3,1],hmstats[4,1])
    @printf(fileID, "iterNo   dtMisfit  mNorm   KEnergy  HEnergy  Accept \n")

    status = zeros(Int, nsamples)
    status[hmcstats.acceptstats] .= 1
    for k = 2: nsamples + 1
        @printf(fileID,"%6d %8.4e %8.4e %8.4e %8.4e %2d\n",k-1,hmstats[1,k],hmstats[2,k],
        hmstats[3,k],hmstats[4,k],status[k-1])
    end
    close(fileID)

end


#-------------------------------------------------------------------------------
function updateStartModel(mtMesh::TensorMesh2D, invParam::InvDataModel)

    # active cell
    activeCell = invParam.activeCell
    bgModel    = invParam.bgModel

    # convert log conductivity into linear conductivity
    strMod = invParam.strModel
    # (sigma, dsigma) = modelTransform(strMod, sigmin, sigmax)
    (sigma, dsigma) = modelTransform(strMod)
    sigma = activeCell * sigma + bgModel
    mtMesh.sigma = copy(sigma)

    return mtMesh, dsigma

end

#-------------------------------------------------------------------------------
include("parallelHMC.jl")
include("readstartupFile.jl")

end # module
