using BenchmarkTools

include("BoseFermiExact.jl")

L = 4
NB = L
NF = 2
NB_max = NB

t = 1.
#t_list = collect(LinRange(0,1,101))
#theta = 44
#UBB = round(sin(π*theta/180); digits = 3)
#UBF = round(cos(π*theta/180); digits = 3)
UBB_list = collect(LinRange(0,1000,101))
#UBF_list = collect(LinRange(200,300,101))
UBF = 0.0

UBF_inf = true
pB = true
pF = true

diag_mode = :Full  #:Sparse or :Full
M = L
tol = 1e-10

read = true
save = false

if UBF_inf
    UBF = 0.
end

println("Generating basis...")
Basis, T = GenerateBFBasis(L, NB, NF; restricted=UBF_inf)
println("Done!, Size=", length(Basis))

foldername = string("results/Data",
                    "_L", L, "_NB", NB, "_NF", NF, "_NBmax", NB_max, 
                    "_t", t, "_UBB", "_UBF", UBF,
                    "_UBFinf", UBF_inf, "_pB", pB, "_pF", pF, "/")
if !isdir(foldername)
    mkdir(foldername)
end

Threads.@threads for UBB in UBB_list
    println("Running UBB=", UBB, "--------------------------------------")
    println("Generating Hamiltonian...")
    HBF = BoseFermiHubbard(Basis, L, NB, NF, NB_max, t, UBB, UBF, UBF_inf, pB, pF, read, save)
    println("Done!")
 
    println("Diagonalizing Hamiltonian...")
    E, Ψ = Diagonalize(HBF; M=M, mode=diag_mode)
    println("Done!")

    println("Saving eigendecomposition...")
    writedlm(string(foldername, "Spectrum",
                    "_L", L, "_NB", NB, "_NF", NF, "_NBmax", NB_max, 
                    "_t", t, "_UBB", UBB, "_UBF", UBF,
                    "_UBFinf", UBF_inf, "_pB", pB, "_pF", pF, ".txt"), E) 
    println("Done!")

    println("Saving densities...")
    for ii in 1:length(E)
        ψ = Ψ[:,ii]
        nᵇ, nᶠ = GetDensities(Basis, ψ, NB_max)
        filename = string(foldername, "ExVal",
                          "_L", L, "_NB", NB, "_NF", NF, "_NBmax", NB_max, 
                          "_t", t, "_UBB", UBB, "_UBF", UBF,
                          "_UBFinf", UBF_inf, "_pB", pB, "_pF", pF, 
                          "_E", ii, ".out")
        open(filename, "w") do io
            writedlm(io, ["# ⟨nᵇ⟩  ⟨nᶠ⟩"]) 
            writedlm(io, hcat(nᵇ, nᶠ)) 
        end
    end
    println("Done!")
end
