#------------------------------------------------------------------------------
# script to perform 2D MT sensitivity test
# Peng Ronghua, Nov., 2021
#
#------------------------------------------------------------------------------
push!(LOAD_PATH, pwd())
push!(LOAD_PATH, joinpath(pwd(),"..","..","src"))
push!(LOAD_PATH, joinpath(pwd(),"..","..","..","MUMPS","src"))
using LinearAlgebra, Printf
import HMCMT
using HMCMT.HMCFileIO
using HMCMT.MTFwdSolver
using HMCMT.MTSensitivity
using HMCMT.HMCUtility
using HMCMT.HMCStruct
using HMCMT.HMCSampler

ENV["OMP_NUM_THREADS"] = 48
ENV["MKL_NUM_THREADS"] = 48
#------------------------------------------------------------------------------


function main(args)
    if isempty(args)
        @isdefined(SyslabCC) || cd(@__DIR__)
        startup = "startupfile"
    else
        startup = args[1]
    end

    println("Reading datafile and modelfile ...\n")
    (mtMesh,mtData,invParam,hmcprior) = readstartupFile(startup)

    println("Performing HMC inversion ...\n")
    cputime = HMCMT.@compat_elapsed (hmcmodel,hmcstats,hmcdata) = runHMCSampler(mtMesh, mtData, invParam, hmcprior)

    println("cputime: $cputime")
    #
    println("Computing Posterior mean model ...\n")
    getPosteriorModel(hmcmodel,mtMesh,invParam,hmcprior)

    println("Output HMC samples ...\n")
    outputHMCSamples(hmcmodel,hmcstats,hmcdata,cputime=cputime)
    println("=======================================")
end

@isdefined(SyslabCC) || main(ARGS)
#

