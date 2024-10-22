using HMCMT: unix_readline
"""
`readEMModel2D(modelfile)` reads model parameters from a EM2D model file.

Input:
    `modelfile:` =::String, name of the model file

Output:
    `em2dMesh` =::TensorMesh2D, occursin model and mesh parameters

"""
function readEMModel2D(modelfile::String)

    fid = try
        open(modelfile, "r")
    catch e
        if e isa Base.IOError
            println("$(modelfile) does not exist, please try again.")
        end
        throw(e)
    end

    yLen = Float64[]
    zLen = Float64[]
    nAir = 0
    airLayer = Float64[]
    gridSize = Int[]
    sigma = Float64[]
    origin = Float64[]
    resType = ""
    ny = 0
    nz = 0

    while !eof(fid)

        cline = strip(unix_readline(fid))

        # ignore comments and empty lines
        while startswith(cline, '#') || isempty(cline) || cline == "\r"
            cline = strip(unix_readline(fid))
        end

        # blocks at y-axis
        if occursin("NY", cline)
            tmp = split(cline)
            ny  = parse(Int, tmp[end])
            nd  = 0
            yLen = zeros(ny)

            while nd < ny
                cline = strip(unix_readline(fid))
                cline = split(cline)
                num = length(cline)
                for i = 1:num
                    nd = nd + 1
                    yLen[nd] = parse(Float64, cline[i])
                end
            end

        # blocks at z-axis
        elseif occursin("NZ", cline)
            tmp = split(cline)
            nz  = parse(Int, tmp[end])
            nd  = 0
            zLen = zeros(nz)

            while nd < nz
                cline = strip(unix_readline(fid))
                cline = split(cline)
                num = length(cline)
                for i = 1:num
                    nd = nd + 1
                    zLen[nd] = parse(Float64, cline[i])
                end
            end

        # air layer, optional
        elseif occursin("NAIR", cline)
            tmp  = split(cline)
            nAir = parse(Int, tmp[end])
            nd   = 0
            airLayer = zeros(nAir)

            while nd < nAir
                cline = strip(unix_readline(fid))
                cline = split(cline)
                num = length(cline)
                for i = 1:num
                    nd = nd + 1
                    airLayer[nd] = parse(Float64, cline[i])
                end
            end

        # resistivity type: resistivity or conductivity
        elseif occursin("Resistivity Type", cline)
            tmp = split(cline)
            resType = tmp[end]

        # model type: linear or logorithmic
        elseif occursin("Model Type", cline)
            tmp = split(cline)
            modType = tmp[end]
            nBlock = ny * nz
            nd = 0

            sigma = zeros(nBlock)
            while nd < nBlock
                cline = strip(unix_readline(fid))
                cline = split(cline)
                num = length(cline)
                for i = 1:num
                    nd = nd + 1
                    sigma[nd] = parse(Float64, cline[i])
                end
            end

            # convert to linear conductivity
            if resType == "Resistivity"
                sigma = 1 ./ sigma
            end
            if modType == "log"
                sigma = exp(sigma)
            end

        # origin
        elseif occursin("Origin", cline)
            tmp = split(cline)
            origin = zeros(2)
            origin[1] = parse(Float64, tmp[end-1])
            origin[2] = parse(Float64, tmp[end])

        end    # if occursin

    end    # while !eof(fid)

    close(fid)

  # append air layers, note that airLayer is from down to up
    if isempty(airLayer)
        airLayer = zeros(0)
    else
        airDep = sum(airLayer)
        zLen   = [reverse(airLayer, 1); zLen]
        origin[2] = origin[2] + airDep
        sigAir = 1e-8
        airMat = ones(ny,nAir) * sigAir
        sigma  = vcat(vec(airMat), sigma)
    end

    nz = length(zLen)
    gridSize = [ny, nz]
    empMat = spzeros(0,0)
    em2dMesh = TensorMesh2D(yLen, zLen, airLayer, gridSize, origin, sigma,
                            empMat, empMat, empMat, empMat, false)

    return em2dMesh

end
