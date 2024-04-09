"""
`writeMT2DData(datafile, mt2dData, predData, dataErr)` output predicated
 MT 2D data.

Input:
    `datafile` =::String
    `datInfo`  =::MTData
    `predData` =::Vector
    `dataErr`  =::Vector

"""
function writeMT2DData(datafile::AbstractString, datInfo::MTData,
                       predData::Array, dataErr::Vector{Float64}=zeros(0))

    datID = open(datafile, "w")

    # data format
    @printf(datID,"%-20s%s\n","Format:","MT2DData_1.0")
    @printf(datID,"# %s\n","file generated in $(Libc.strftime(time()))")

    # receiver location
    rxLoc = datInfo.rxLoc
    nr = size(rxLoc, 1)
    @printf(datID,"%-25s %4d\n","Receiver Location (m):", nr)
    @printf(datID,"# %5s %5s\n","Y","Z")
    for i = 1:nr
        @printf(datID,"%12.2f %12.2f\n",rxLoc[i,1], rxLoc[i,2])
    end

    # frequencies
    freqs = datInfo.freqs
    nF = length(freqs)
    @printf(datID,"%-20s%3d\n","Frequencies (Hz):",nF)
    for i = 1:nF
        @printf(datID,"%8.4e\n",freqs[i])
    end

    # data name
    dataType = datInfo.dataType
    @printf(datID,"%-12s %12s\n","DataType:",dataType)

    # data type
    dataComp = datInfo.dataComp
    nDT = length(dataComp)
    @printf(datID,"%-15s %d\n","DataComp:",nDT)
    for i = 1:nDT
        @printf(datID,"%4s\n",dataComp[i])
    end

    # data block
    # dataComp includes:
    #   RhoXY, PhsXY, RhoYX, PhsYX, ZXY, ZYX
    if isempty(dataErr)
        dataErr = abs.(predData) * 0.03
    elseif length(dataErr) .== 1
        dataErr = abs.(predData) * dataErr[1]
    end

    rxID   = datInfo.rxID
    freqID = datInfo.freqID
    dtID   = datInfo.dtID
    nData  = size(predData,1)

    @printf(datID,"%-15s %d\n","Data Block:",nData)

    if eltype(predData) <: Complex     # iseltype(predData, Complex)
        # freqID   recID   DataTpye  RealPart ImagPart DataError
        @printf(datID,"# %6s %6s %10s %10s %15s %12s\n","FreqNo.","RxNo.",
                "dataComp","RealValue","ImagValue","Error")
        for i = 1:nData
            @printf(datID,"%5d %6d %8d %15.6e %15.6e %15.6e\n", freqID[i],
                    rxID[i], dtID[i], real(predData[i]), imag(predData[i]), dataErr[i])
        end
    else
        # freqID   recID   DataTpye  RealPart ImagPart DataError
        @printf(datID,"# %6s %6s %10s %10s %12s\n","FreqNo.","RxNo.",
                "dataComp","RealValue","Error")
        for i = 1:nData
            @printf(datID,"%5d %6d %8d %15.6e %15.6e\n", freqID[i],
                    rxID[i], dtID[i], predData[i], dataErr[i])
        end
    end

    close(datID)

end
