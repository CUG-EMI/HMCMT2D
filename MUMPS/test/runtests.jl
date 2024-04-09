push!(LOAD_PATH, pwd())
push!(LOAD_PATH, joinpath(pwd(),"..","src"))
using Test

println("test MUMPS")
@testset "MUMPS" begin
    @testset "DivGrad" begin
        include("testDivGrad.jl")
    end
    @testset "Two systems" begin
        include("testTwoSystem.jl")
    end
    println("Done!")
end
