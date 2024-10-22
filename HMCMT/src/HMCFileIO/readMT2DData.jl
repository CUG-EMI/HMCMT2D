using HMCMT: unix_readline
#-------------------------------------------------------------------------------
"""
`readMT2DData(datafile)` reads MT2D data file.

Input:
    datafile =::String, data file name

Output:
    datInfo  =::MTData
    obsData  =::Array
    dataErr  =::Array

"""
function readMT2DData(datafile::String)

    fid = try
        open(datafile, "r")
    catch e
        if e isa Base.IOError
            println("$(datafile) does not exist, please try again.")
        end
        throw(e)
    end

    rxLoc = zeros(Float64, 0, 0)
    rxID  = Int[]
    freqID = Int[]
    dtID   = Int[]
    obsData = Float64[]::Union{Vector{ComplexF64}, Vector{Float64}}
    dataErr = Float64[]
    dataType  = ""
    dataComp  = String[]
    freqs = Float64[]
    compTE = false
    compTM = false
    isComplex = false
    nr  = 0; nDt = 0; nf  = 0
    while !eof(fid)

        cline = strip(unix_readline(fid))

        # ignore all comments: empty line, or line preceded with #
        while startswith(cline, '#') || isempty(cline) || cline == "\r"
            cline = strip(unix_readline(fid))
        end

        # data format
        if occursin("Format", cline)
            tmp = split(cline)
            format = tmp[2]

        # receiver location
        elseif occursin("Receiver Location", cline)
            tmp = split(cline)
            nr  = parse(Int, tmp[end])
            nd  = 0
            rxLoc = zeros(nr, 2)
            while nd < nr
                cline = strip(unix_readline(fid))
                while startswith(cline, '#') || isempty(cline) || cline == "\r"
                    cline = strip(unix_readline(fid))
                end
                cline = split(cline)
                nd = nd + 1
                for j = 1:2
                    rxLoc[nd,j] = parse(Float64, cline[j])
                end
            end

        # frequencies
        elseif occursin("Frequencies", cline)
            tmp = split(cline)
            nf  = parse(Int, tmp[end])
            nd  = 0
            freqs = zeros(nf)
            while nd < nf
                cline = strip(unix_readline(fid))
                while startswith(cline, '#') || isempty(cline) || cline == "\r"
                    cline = strip(unix_readline(fid))
                end
                #cline = split(cline)
                nd = nd + 1
                freqs[nd] = parse(Float64, cline)
            end

        # data type
        elseif occursin("DataType", cline)
            tmp = split(cline)
            dataType = string(tmp[end])
            if dataType != "Impedance" && (dataType != "Rho_Pha")
                error("$(dataType) is not supported.")
            end

            if dataType == "Impedance"
                isComplex = true
            else
                isComplex = false
            end

        # data components
        elseif occursin("DataComp", cline)
            tmp = split(cline)
            nDt = parse(Int, tmp[end])
            dataComp = Array{String}(undef, nDt)
            nd = 0
            while nd < nDt
                cline = strip(unix_readline(fid))
                nd = nd + 1
                dataComp[nd] = cline
            end

        # data block
        elseif occursin("Data Block", cline)
            tmp   = split(cline)
            nData = parse(Int, tmp[end])
            nd    = 0
            rxID  = zeros(Int, nData)
            dtID  = zeros(Int, nData)
            freqID = zeros(Int, nData)
            if isComplex
                obsData = zeros(ComplexF64, nData)
            else
                obsData = zeros(Float64, nData)
            end
            dataErr = zeros(nData)

            while nd < nData
                cline = strip(unix_readline(fid))
                while startswith(cline, '#') || isempty(cline) || cline == "\r"
                    cline = strip(unix_readline(fid))
                end
                cline = split(cline)
                nd = nd + 1
                freqID[nd] = parse(Int, cline[1])
                rxID[nd]   = parse(Int, cline[2])
                dtID[nd]   = parse(Int, cline[3])
                if isComplex
                    obsData[nd] = parse(Float64,cline[4]) + parse(Float64, cline[5])*1im
                    dataErr[nd] = parse(Float64,cline[6])
                else
                    obsData[nd] = parse(Float64, cline[4])
                    dataErr[nd] = parse(Float64, cline[5])
                end

            end

        end

    end


    # figure out what modes are required
    for j in 1:length(dataComp)
        if occursin("XY", dataComp[j])
            compTE = true
            break
        end
    end

    for j in 1:length(dataComp)
        if occursin("YX", dataComp[j])
            compTM = true
            break
        end
    end

    # define dataID
    dataID = zeros(Bool,nDt,nr,nf)
    for k = 1:length(obsData)
        ridx = rxID[k]
        fidx = freqID[k]
        didx = dtID[k]
        dataID[didx,ridx,fidx] = true
    end
    dataID = vec(dataID)


    datInfo = MTData(rxLoc, freqs, dataType, dataComp, rxID, freqID, dtID, dataID, compTE, compTM)

    return ReadMT2DDataRet(datInfo, obsData, dataErr)

end

struct ReadMT2DDataRet
    a::MTData
    b::Union{Vector{ComplexF64}, Vector{Float64}}
    c::Vector{Float64}
end
