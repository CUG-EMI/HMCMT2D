"""
`writeMT2DModel(modelfile, model2d)` writes model parameters into a MT2D model file.

Input:
    `modelfile:` =::String, name of the model file
    `em2dMesh` =::TensorMesh2D, contains model and mesh parameters

Output:

"""
function writeEMModel2D(modelfile::String, em2dMesh::TensorMesh2D)

    fid = open(modelfile, "w")

    # model format
    @printf(fid,"%-18s %s\n", "#Format:", "EMModel2DFile")
    @printf(fid,"%-18s %s\n","#Description:","file generated in $(Libc.strftime(time()))")

    ny = length(em2dMesh.yLen)
    nz = length(em2dMesh.zLen)

    # y block
    @printf(fid,"%-6s %4d\n","NY:", ny)
    for i = 1:ny
        @printf(fid,"%10.2f", em2dMesh.yLen[i])
        if mod(i,8) == 0;  @printf(fid,"\n");  end
    end
    if mod(ny,8) != 0;  @printf(fid,"\n");  end


    # air layer
    nAir = 0
    if !isempty(em2dMesh.airLayer)
        nAir = length(em2dMesh.airLayer)
        @printf(fid,"%-6s %4d\n","NAIR:", nAir)
        for i = 1:nAir
            @printf(fid,"%12.2f", em2dMesh.airLayer[i])
            if mod(i,8) == 0;  @printf(fid,"\n");  end
        end
        if mod(nAir,8) != 0;  @printf(fid,"\n");  end
    end

    # z block
    @printf(fid,"%-6s %4d\n","NZ:",  nz - nAir)
    for i = nAir+1:nz
        @printf(fid,"%10.2f", em2dMesh.zLen[i])
        if mod(i-nAir,8) == 0;  @printf(fid,"\n");  end
    end
    if mod(nz-nAir,8) != 0;  @printf(fid,"\n");  end


    # conductivity
    sigma = copy(em2dMesh.sigma)
    airCell = ny * nAir
    sigma   = sigma[airCell+1:end]

    sigma = reshape(sigma, ny, nz-nAir)
    resType = "Conductivity"
    modType = "Linear"

    @printf(fid,"%-18s %s\n","Resistivity Type:",resType)
    @printf(fid,"%-18s %s\n","Model Type:", modType)

    for k = 1:nz-nAir
        for j = 1:ny
            @printf(fid,"%4.2e ",sigma[j,k])
        end
        @printf(fid,"\n")
    end

    # origin
    origin = copy(em2dMesh.origin)
    if nAir > 0
        airDep = sum(em2dMesh.airLayer)
        origin[2] -= airDep
    end
    @printf(fid, "%-15s %4.2e %4.2e","Origin (m):",
            origin[1], origin[2])

    close(fid)

end
