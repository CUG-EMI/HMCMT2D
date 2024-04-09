export parallelHMCSampler

#------------------------------------------------------------------------------
"""
    `parallelHMCSampler(mtMesh, mtData, invParam, hmcprior, pids)`

run multiple HMC chains in parallel.

"""
function parallelHMCSampler(mtMesh::TensorMesh2D,
                            mtData::MTData,
                            invParam::InvDataModel,
                            hmcprior::HMCPrior,
                            pids::Vector)

    #
    np = length(pids)
    hmcmodel = Array{Any}(undef, np)
    hmcstats = Array{Any}(undef, np)
    hmcdata  = Array{Any}(undef, np)
    cputime = Array{Float64}(undef, np)
    # run multiple chains in parallel
    i = 1
    nextidx() = (idx = i; i+=1; idx)
    @sync begin

        for p = pids
            @async begin
                while true
                    idx = nextidx()
                    if idx > np
                        break
                    end
                cputime[idx] = @elapsed (hmcmodel[idx],hmcstats[idx],hmcdata[idx]) =
                remotecall_fetch(runHMCSampler,p,mtMesh,mtData,invParam,hmcprior)
                end
            end # @async
        end # p

    end # @sync

    # output chain samples
    for k = 1:np
        outputHMCSamples(hmcmodel[k], hmcstats[k], hmcdata[k], ichain=k, cputime=cputime[k])
    end

    return hmcmodel, hmcstats, hmcdata

end
