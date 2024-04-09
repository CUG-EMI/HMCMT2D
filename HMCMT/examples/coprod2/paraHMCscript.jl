#------------------------------------------------------------------------------
# script to perform Hamiltonian Monte Carlo sampling for 2D MT problem.
# (c) Peng Ronghua, April, 2023
#
#------------------------------------------------------------------------------
@everywhere push!(LOAD_PATH, pwd())
@everywhere push!(LOAD_PATH, joinpath(pwd(),"..","..","src"))
@everywhere push!(LOAD_PATH, joinpath(pwd(),"..","..","..","MUMPS","src"))
@everywhere begin
    using HMCMT.HMCFileIO
    using HMCMT.MTFwdSolver
    using HMCMT.MTSensitivity
    using HMCMT.HMCUtility
    using HMCMT.HMCStruct
    using HMCMT.HMCSampler
end

#------------------------------------------------------------------------------
@everywhere begin
    printstyled("read datafile and modelfile ...\n", color=:blue)
    startup = "startupfile"
    (mtMesh,mtData,invParam,hmcprior) = readstartupFile(startup)
end

#
printstyled("perform HMC sampling in parallel ...\n", color=:blue)
pids = workers()
@time (results, hmcstats) = parallelHMCSampler(mtMesh,mtData,invParam,hmcprior,pids)

println("=======================================")
