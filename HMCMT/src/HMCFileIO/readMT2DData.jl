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

    if isfile(datafile)
        fid = open(datafile, "r")
    else
        error("$(datafile) does not exist, please try again.")
    end

    rxLoc = []
    rxID  = []
    freqID = []
    dtID   = []
    obsData = []
    dataErr = []
    dataType  = []
    dataComp  = []
    freqs = []
    compTE = false
    compTM = false
    isComplex = false
    nr  = []; nDt = []; nf  = []
    while !eof(fid)

        cline = strip(readline(fid))

        # ignore all comments: empty line, or line preceded with #
        while cline[1] == '#' || isempty(cline)
            cline = strip(readline(fid))
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
                cline = strip(readline(fid))
                while cline[1] == '#' || isempty(cline)
                    cline = strip(readline(fid))
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
                cline = strip(readline(fid))
                while cline[1] == '#' || isempty(cline)
                    cline = strip(readline(fid))
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
                cline = strip(readline(fid))
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
                cline = strip(readline(fid))
                while cline[1] == '#' || isempty(cline)
                    cline = strip(readline(fid))
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

    return datInfo, obsData, dataErr

end
