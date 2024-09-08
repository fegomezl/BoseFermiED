using BenchmarkTools

include("BoseFermiExact.jl")

L = 9
NB = L
NF = 2
NB_max = NB

t = 1.
UBB = 1.
UBF = 1.

UBF_inf = true
pB = true
pF = true

diag_mode = :Sparse
M = 3*L
tol = 1e-10

read = false
save = true

if UBF_inf
    UBF = 0.
end

println("Generating basis...")
Basis, T = GenerateBFBasis(L, NB, NF; restricted=UBF_inf)
println("Done!, Size=", length(Basis))

println("Generating Hamiltonian...")
HBF = BoseFermiHubbard(Basis, L, NB, NF, NB_max, t, UBB, UBF, UBF_inf, pB, pF, read, save)
println("Done!")

println("Diagonalizing Hamiltonian...")
E, Ψ = Diagonalize(HBF; M=M, mode=diag_mode)
println("Done!")

println("Saving eigendecomposition...")
writedlm(string("results/Spectrum",
                "_L", L, "_NB", NB, "_NF", NF, "_NBmax", NB_max, 
                "_t", t, "_UBB", UBB, "_UBF", UBF,
                "_UBFinf", UBF_inf, "_pB", pB, "_pF", pF, ".txt"), 
         E)
println("Done!")

println("Saving densities...")
for ii in 1:M
    ψ = Ψ[:,ii]
    nᵇ, nᶠ = GetDensities(Basis, ψ, NB_max)
    filename = string("results/Spectrum",
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
