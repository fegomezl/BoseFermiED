using LinearAlgebra
using SparseArrays
using ArnoldiMethod

using Combinatorics
using BenchmarkTools
using DelimitedFiles

BoseBasisSize(L::Int, N::Int)  = binomial(L+N-1, N)
FermiBasisSize(L::Int, N::Int) = binomial(L, N)

function MultipleOccupancy(x::Vector{Int}, N_max::Int)::Bool
    for n in x
        if n > N_max
            return false
        end
    end
    return true
end

function GenerateBasis(L::Int, N::Int, N_max::Int = N)::Tuple{Vector{Vector{Int64}}, BitVector}
    Basis = Array{Vector{Int}}(undef, BoseBasisSize(L, N))

    n = 1
    x = zeros(Int, L)
    x[1] = N
    Basis[n] = copy(x)
    
    while x[L] ≠ N
        
        n += 1
        x[L] = N
        
        kk = L-1
        while x[kk] == 0
            kk -= 1
        end
        
        x[kk] -= 1
        x[kk+1] = N-sum(x[1:kk])
        x[kk+2:L] .= 0
        Basis[n] = copy(x)
    end
    

    if N_max < N
        T = MultipleOccupancy.(Basis, N_max)
        filter!(x->MultipleOccupancy(x, N_max), Basis)
        return Basis, T
    else
        T = MultipleOccupancy.(Basis, N)
        return Basis, T
    end
end

function BasisTransformation(x::Vector{TV}, T::BitVector)::Vector{TV} where TV
    y = Array{TV}(undef, length(T))
    jj = 0
    for ii in eachindex(T)
        if T[ii]
            jj += 1
            y[ii] = x[jj]
        else
            y[ii] = 0
        end
    end
    return y
end

function GenerateBFBasis(L::Int, NB::Int, NF::Int; NB_max::Int = NB, restricted::Bool = false)::Tuple{Vector{Vector{Int64}}, BitVector}
    BoseBasis, TB = GenerateBasis(L, NB, NB_max)
    FermiBasis, TF = GenerateBasis(L, NF, 1)
    BoseFermiBasis = Array{Vector{Int}}(undef, length(BoseBasis)*length(FermiBasis))

    n = 0
    for xB in BoseBasis
        for xF in FermiBasis
            n += 1
            BoseFermiBasis[n] = (NB_max+1)*xF+xB  # nF*(NBmax+1)+nB is a one-to-one mapping from (nF,nB) to integers
        end
    end

    if restricted
        TBF = MultipleOccupancy.(BoseFermiBasis, NB_max+1)
        filter!(x->MultipleOccupancy(x, NB_max+1), BoseFermiBasis)
        return BoseFermiBasis, TBF
    else
        TBF = MultipleOccupancy.(BoseFermiBasis, 2*NB_max+1)
        return BoseFermiBasis, TBF
    end
end

function GenerateTag(x::Vector{Int})::Float64
    T = 0
    for ii in eachindex(x)
        T += log((100*ii+3))*x[ii]
    end
    return T
end

function AddBosonHopping!(XYV::Vector{Tuple{Int64, Int64, Float64}}, N::Int, n::Int, x::Vector{Int}, ii::Int, jj::Int; Tags::Vector{Float64}, t::Float64, NB_max::Int)
    y = copy(x)
    y[ii] -= 1
    y[jj] += 1
    Tag = GenerateTag(y)
    m_list = findall(T -> T == Tag, Tags)
    for m in m_list
        XYV[N] = (n, m, t*√((x[ii]%(NB_max+1))*(y[jj]%(NB_max+1))))
    end
end

function AddFermionHopping!(XYV::Vector{Tuple{Int64, Int64, Float64}}, N::Int, n::Int, x::Vector{Int}, ii::Int, jj::Int; Tags::Vector{Float64}, t::Float64, NB_max::Int)
    y = copy(x)
    y[ii] -= NB_max+1
    y[jj] += NB_max+1
    Tag = GenerateTag(y)
    m_list = findall(T -> T == Tag, Tags)
    for m in m_list
        XYV[N] = (n, m, -t)
    end
end

function AddBosonInteraction!(XYV::Vector{Tuple{Int64, Int64, Float64}}, N::Int, n::Int, x::Vector{Int}, ii::Int; UBB::Float64, NB_max::Int)
    XYV[N] = (n, n, UBB*(x[ii]%(NB_max+1))*((x[ii]%(NB_max+1)-1)))
end

function AddBoseFermiInteraction!(XYV::Vector{Tuple{Int64, Int64, Float64}}, N::Int, n::Int, x::Vector{Int}, ii::Int; UBF::Float64, NB_max::Int)
    XYV[N] = (n, n, UBF*(x[ii]%(NB_max+1))*(x[ii]÷(NB_max+1)))
end

function BoseFermiHubbard(Basis::Vector{Vector{Int64}}, L::Int, NB::Int, NF::Int, NB_max::Int, t::Float64, UBB::Float64, UBF::Float64, UBF_inf::Bool=false, pB::Bool=false, pF::Bool=false, read::Bool = false, save::Bool = false)::SparseMatrixCSC{Float64, Int64}
    D = length(Basis)
    Tags = GenerateTag.(Basis)

    if save

        XYV = fill((1,1,0.0), 4*L*D)

        Threads.@threads for n in 1:D
            
            N = 4*L*(n-1)
            x = Basis[n]

            for ii in 1:L-1
                N += 1
                AddBosonHopping!(XYV, N, n, x, ii, ii+1; Tags, t, NB_max)
                N += 1
                AddBosonHopping!(XYV, N, n, x, ii+1, ii; Tags, t, NB_max) 
                N += 1
                AddFermionHopping!(XYV, N, n, x, ii, ii+1; Tags, t, NB_max)
                N += 1
                AddFermionHopping!(XYV, N, n, x, ii+1, ii; Tags, t, NB_max)
            end
            if pB
                N += 1
                AddBosonHopping!(XYV, N, n, x, 1, L; Tags, t, NB_max)
                N += 1
                AddBosonHopping!(XYV, N, n, x, L, 1; Tags, t, NB_max)
            else
                N += 2
            end
            if pF
                N += 1
                AddFermionHopping!(XYV, N, n, x, 1, L; Tags, t, NB_max)
                N += 1
                AddFermionHopping!(XYV, N, n, x, L, 1; Tags, t, NB_max)
            else
                N += 2
            end
        end
        writedlm(string("hamiltonian/Hamiltonian_L", L, "_NB", NB, "_NF", NF, "_NBmax", NB_max, "_UBFinf", UBF_inf, "_pB", pB, "_pF", pF, ".txt"), XYV)

        H = sparse(getfield.(XYV, 1), getfield.(XYV, 2), getfield.(XYV, 3), D, D)
        dropzeros!(H)
        return H

    elseif read

        XYV_read = readdlm(string("hamiltonian/Hamiltonian_L", L, "_NB", NB, "_NF", NF, "_NBmax", NB_max, "_UBFinf", UBF_inf, "_pB", pB, "_pF", pF, ".txt"))
        XYV_read = Vector{Tuple{Int64, Int64, Float64}}(tuple.(eachcol(XYV_read)...))
        for ii in 1:length(XYV_read)
            x, y, v = XYV_read[ii]
            XYV_read[ii] = (x, y, t*v)
        end

        XYV = fill((1,1,0.0), 2*L*D)

        Threads.@threads for n in 1:D
            
            N = 2*L*(n-1)
            x = Basis[n]

            for ii in 1:L
                N += 1
                AddBosonInteraction!(XYV, N, n, x, ii; UBB, NB_max)
                N += 1
                AddBoseFermiInteraction!(XYV, N, n, x, ii; UBF, NB_max)
            end
        end
        XYV = [XYV; XYV_read]

        H = sparse(getfield.(XYV, 1), getfield.(XYV, 2), getfield.(XYV, 3), D, D)
        dropzeros!(H)
        return H

    else

        XYV = fill((1,1,0.0), 6*L*D)

        Threads.@threads for n in 1:D
            
            N = 6*L*(n-1)
            x = Basis[n]

            for ii in 1:L-1
                N += 1
                AddBosonHopping!(XYV, N, n, x, ii, ii+1; Tags, t, NB_max)
                N += 1
                AddBosonHopping!(XYV, N, n, x, ii+1, ii; Tags, t, NB_max) 
                N += 1
                AddFermionHopping!(XYV, N, n, x, ii, ii+1; Tags, t, NB_max)
                N += 1
                AddFermionHopping!(XYV, N, n, x, ii+1, ii; Tags, t, NB_max)
            end
            if pB
                N += 1
                AddBosonHopping!(XYV, N, n, x, 1, L; Tags, t, NB_max)
                N += 1
                AddBosonHopping!(XYV, N, n, x, L, 1; Tags, t, NB_max)
            else
                N += 2
            end
            if pF
                N += 1
                AddFermionHopping!(XYV, N, n, x, 1, L; Tags, t, NB_max)
                N += 1
                AddFermionHopping!(XYV, N, n, x, L, 1; Tags, t, NB_max)
            else
                N += 2
            end

            for ii in 1:L
                N += 1
                AddBosonInteraction!(XYV, N, n, x, ii; UBB, NB_max)
                N += 1
                AddBoseFermiInteraction!(XYV, N, n, x, ii; UBF, NB_max)
            end
        end
        
        H = sparse(getfield.(XYV, 1), getfield.(XYV, 2), getfield.(XYV, 3), D, D)
        dropzeros!(H)
        return H

    end
end

function Diagonalize(HBF::SparseMatrixCSC{Float64, Int64}; M::Int=1, tol::Float64=1e-8, mode=:Sparse)::Tuple{Vector{Float64}, Matrix{Float64}}
    if mode == :Sparse
        decomp, history = partialschur(HBF, nev=M, tol=tol, which=SR())
        E = Real.(decomp.eigenvalues)
        Ψ = Matrix(decomp.Q)
        return E, Ψ
    elseif mode == :Full
        full_HBF = Symmetric(Matrix(HBF))
        E, Ψ = eigen(full_HBF)
        return E, Ψ
    end
end

# --------------- Measurements -----------------------------------

function GetDensities(Basis::Vector{Vector{Int64}}, ψ::Vector{T}, NB_max::Int)::Tuple{Vector{Float64}, Vector{Float64}} where T
    L = length(Basis[1])
    nᵇ = zeros(Float64, L)
    nᶠ = zeros(Float64, L)
    for (ii, x) in enumerate(Basis)
        nᵇ += @.(x%(NB_max+1))*ψ[ii]^2
        nᶠ += @.(x÷(NB_max+1))*ψ[ii]^2
    end
    return nᵇ, nᶠ
end
