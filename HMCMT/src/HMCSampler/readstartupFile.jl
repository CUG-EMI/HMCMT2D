export readstartupFile
using HMCMT: unix_readline;

#-------------------------------------------------------------------------------
function readstartupFile(startupfile::String)

    #
    fid = try
        open(startupfile, "r")
    catch e
        if e isa Base.IOError
            println("$(startupfile) does not exist, please try again.")
        end
        throw(e)
    end

    datafile  = ""
    modelfile = ""
    sigmin = 0.0
    sigmax = 0.0
    sigfix = [1e-8]
    hmcprior = initHMCPrior()

    while !eof(fid)
        cline = strip(unix_readline(fid))
        println(repr(cline))

        # ignore all comments: empty line, or line preceded with #
        while startswith(cline, '#') || isempty(cline) || cline == "\r"
            cline = strip(unix_readline(fid))
            continue
        end

        # data and start model filename
        if occursin("datafile:", cline)
            tmp = split(cline)
            datafile = string(tmp[end])

        elseif occursin("modelfile:", cline)
            tmp = split(cline)
            modelfile = string(tmp[end])

        # markov chain parameter
        elseif occursin("burninsamples:", cline)
            tmp = split(cline)
            hmcprior.burninsamples = parse(Int, tmp[end])

        elseif occursin("totalsamples:", cline)
            tmp = split(cline)
            hmcprior.totalsamples = parse(Int, tmp[end])

        # proposal parameter
        elseif occursin("resistivity:", cline)
            tmp = split(cline)
            rhomin = parse(Float64, tmp[end-2])
            rhomax = parse(Float64, tmp[end-1])
            rhostd = parse(Float64, tmp[end])
            #
            sigmin = 1/rhomax
            sigmax = 1/rhomin
            hmcprior.sigBounds = [sigmin, sigmax]
            hmcprior.sigmastd  = (log(sigmax) - log(sigmin))*0.05

        elseif occursin("fixedresistivity:", cline)
            tmp = split(cline)
            tmpa = parse(Float64, tmp[end])
            append!(sigfix, tmpa)

        elseif occursin("timeinterval:", cline)
            tmp = split(cline)
            hmcprior.dt = parse(Float64, tmp[end])

        elseif occursin("timestep:", cline)
            tmp = split(cline)
            hmcprior.timestep[1] = parse(Int, tmp[end-1])
            hmcprior.timestep[2] = parse(Int, tmp[end])

        elseif occursin("linearsolver:", cline)
            tmp = split(cline)
            hmcprior.linearSolver = string(tmp[end])

        elseif occursin("masstype:", cline)
            tmp = split(cline)
            hmcprior.massType = string(tmp[end])

        elseif occursin("smoothparameter:", cline)
            tmp = split(cline)
            hmcprior.regParam = parse(Float64, tmp[end])

        end
    end

    #
    println("Reading data file $(datafile) ...")
    readMTRet = readMT2DData(datafile)
    mtData = readMTRet.a
    obsData = readMTRet.b
    dataErr = readMTRet.c

    println("Reading model file $modelfile ...")
    mtMesh =  readEMModel2D(modelfile)
    if !mtMesh.setup
        setupTensorMesh2D!(mtMesh)
    end

    #
    println("setup inverse data and model ...")
    invParam = setupInverseDataModel(mtMesh,sigfix,sigmin,sigmax,obsData,dataErr)

    return mtMesh, mtData, invParam, hmcprior

end
