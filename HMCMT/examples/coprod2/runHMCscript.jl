#------------------------------------------------------------------------------
# script to perform 2D MT sensitivity test
# Peng Ronghua, Nov., 2021
#
#------------------------------------------------------------------------------
# push!(LOAD_PATH,"/home/pc/pengrh/juliaRepo/smartEMv10/")
push!(LOAD_PATH, pwd())
push!(LOAD_PATH, joinpath(pwd(),"..","..","src"))
push!(LOAD_PATH, joinpath(pwd(),"..","..","..","MUMPS","src"))
using LinearAlgebra, Printf
using HMCMT.HMCFileIO
using HMCMT.MTFwdSolver
using HMCMT.MTSensitivity
using HMCMT.HMCUtility
using HMCMT.HMCStruct
using HMCMT.HMCSampler

ENV["OMP_NUM_THREADS"] = 48
ENV["MKL_NUM_THREADS"] = 48
#------------------------------------------------------------------------------
printstyled("Reading datafile and modelfile ...\n", color=:blue)
startup = "startupfile"
(mtMesh,mtData,invParam,hmcprior) = readstartupFile(startup)

#
printstyled("Performing HMC inversion ...\n", color=:blue)
cputime = @elapsed (hmcmodel,hmcstats,hmcdata) = runHMCSampler(mtMesh, mtData, invParam, hmcprior)

#
printstyled("Computing Posterior mean model ...\n", color=:blue)
getPosteriorModel(hmcmodel,mtMesh,invParam,hmcprior)

printstyled("Output HMC samples ...\n", color=:blue)
outputHMCSamples(hmcmodel,hmcstats,hmcdata,cputime=cputime)

#
println("=======================================")
