import numpy as np
import glob

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def centraldiff(v):
    dv = np.zeros(len(v))
    dv[0] = v[1]-v[0]
    for ii in range(0,len(v)-2):
        dv[ii+1] = (v[ii+2]-v[ii])/2
    dv[-1] = v[-1]-v[-2]
    return dv
    
def derivate(y, x):
    dx = centraldiff(x)
    dy = centraldiff(y)
    return [dy[ii]/dx[ii] for ii in range(len(x))]

def get_Log(folder, parameter):
    filepath = glob.glob(folder+"*.log")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-4].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                param = float(x[n:])
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]
    
    Sweep = []
    Time = []
    Energy = []
    Entropy = []
    Energy_error = []
    Entropy_error = []
    
    for fp in filepath:
        Out = genfromlog(fp)
        Sweep.append(Out[-1,0])
        Time.append(Out[-1,1])
        Energy.append(Out[-1,2])
        Entropy.append(Out[-1,3])
        Energy_error.append(Out[-1,4])
        Entropy_error.append(Out[-1,5])
    
    Sweep = np.array(Sweep)
    Time = np.array(Time)
    Energy = np.array(Energy)
    Entropy = np.array(Entropy)
    Energy_error = np.array(Energy_error)
    Entropy_error = np.array(Entropy_error)

    Log = {"Sweep": Sweep,
           "Time": Time,
           "Energy": Energy,
           "Entropy": Entropy,
           "Energy_error": Energy_error,
           "Entropy_error": Entropy_error}

    return Param, Log

def get_mu(NB, Energy):
    mu = [np.NaN]
    for ii in range(1, len(NB)):
        if NB[ii-1] == NB[ii]-1:
            mu.append(Energy[ii]-Energy[ii-1])
        else:
            mu.append(np.NaN)
    return mu

def get_Gap(files, parameter):
    
    Param_list = []
    E_list = []

    for file in files:
        Param, Log = get_Log(file, parameter)
        Param_list.append(Param)
        E_list.append(Log["Energy"])

    Gap = []
    for ii_0, Param_0 in enumerate(Param_list[1]):
        E_0 = E_list[1][ii_0]
        
        if len(np.where(Param_list[0] == Param_0)[0]) != 0:
            ii_m = np.where(Param_list[0] == Param_0)[0][0]
            E_m = E_list[0][ii_m]
        else:
            E_m = np.NaN
            print("{}={} lower bound not found!".format(parameter, Param_0))
        
        if len(np.where(Param_list[2] == Param_0)[0]) != 0:
            ii_p = np.where(Param_list[2] == Param_0)[0][0]
            E_p = E_list[2][ii_p]
        else:
            E_p = np.NaN
            print("{}={} upper bound not found!".format(parameter, Param_0))

        Gap.append((E_m+E_p)-2*E_0)

    return Param_list[1], np.array(Gap)

def get_Density(folder, parameter):
    filepath = glob.glob(folder+"*.out")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-4].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                try: 
                    param = float(x[n:])
                except:
                    None
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]
    
    Nb = []
    Nf = []
    NbNb = []
    NbNf = []
    S = []
    
    for fp in filepath:
        Out = np.genfromtxt(fp)
        Nb.append(Out[:,1])
        Nf.append(Out[:,2])
        NbNb.append(Out[:,3])
        NbNf.append(Out[:,4])
        S.append(Out[:,5])
    
    Nb = np.array(Nb)
    Nf = np.array(Nf)
    NbNb = np.array(NbNb)
    NbNf = np.array(NbNf)
    S = np.array(S)

    Density = {"Nb": Nb,
               "Nf": Nf,
               "NbNb": NbNb,
               "NbNf": NbNf,
               "S": S}

    return Param, Density

def get_Overlap(folder):
    filepath = glob.glob(folder+"/**/*.out")
    filepath = np.concatenate((filepath, glob.glob(folder+"/**/*.aux")))
    file = [fp.split("/")[-1] for fp in filepath]
    
    UBB = np.zeros(len(file))
    UBF = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-4].split("_")
        ubb = 0.
        ubf = 0.
        for x in fs:
            if x[0:3] == "UBB":
                ubb = float(x[3:])
            if x[0:3] == "UBF":
                ubf = float(x[3:])
        UBB[ii] = ubb
        UBF[ii] = ubf

    zipU = zip(UBB, UBF, file, filepath)
    sort_pairs = sorted(zipU)
    tuples = zip(*sort_pairs)
    UBB, UBF, file, filepath = [ list(tuple) for tuple in  tuples]
    UBB = np.array(UBB)
    UBF = np.array(UBF)
    
    Overlap = []
    
    for fp in filepath:
        if fp[-4:] == ".out":
            OUT = np.genfromtxt(fp)
            
            Nb = np.array(OUT[:,1])
            Nf = np.array(OUT[:,2])
            
            Overlap.append(sum(Nb*Nf))
        else:
            Overlap.append(np.NaN)
    
    Overlap = np.array(Overlap)

    return UBB, UBF, Overlap

def lin_to_square(A):
    n2 = A.size
    n = int(np.sqrt(n2))
    if (n*n == n2):
        return A.reshape((n,n))
        
def get_Correlation(file):
    try:
        Corr = np.genfromtxt(file)
    except:
        return None, None, None
    
    x          = lin_to_square(Corr[:,0])
    y          = lin_to_square(Corr[:,1])
    BtB        = lin_to_square(Corr[:,2])
    NbNb       = lin_to_square(Corr[:,3])
    CtC        = lin_to_square(Corr[:,4])
    NfNf       = lin_to_square(Corr[:,5])
    BtCtBC     = lin_to_square(Corr[:,6])
    NbNf       = lin_to_square(Corr[:,7])
    BtCCtB     = lin_to_square(Corr[:,8])

    Corr = {"BtB":BtB,
            "NbNb":NbNb,
            "CtC":CtC,
            "NfNf":NfNf,
            "BtCtBC":BtCtBC,
            "NbNf":NbNf,
            "BtCCtB":BtCCtB}

    return Corr


def get_Distance(folder, parameter, periodic=False):
    filepath = glob.glob(folder+"*.cout")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-5].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                param = float(x[n:])
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]
    
    DBB   = np.zeros(len(Param))
    DFF   = np.zeros(len(Param))
    DBF   = np.zeros(len(Param))

    for (ii, fp) in enumerate(filepath):
        Corr = get_Correlation(fp)

        NB = sum(np.diag(Corr["BtB"]))
        NF = sum(np.diag(Corr["CtC"]))
            
        for ix in range(len(Corr["BtB"])):
            for iy in range(len(Corr["BtB"])):
                if periodic:
                    DBB[ii]   += min(np.abs(ix-iy), len(x)-np.abs(ix-iy))*Corr["NbNb"][ix, iy]
                    DFF[ii]   += np.abs(ix-iy)*Corr["NfNf"][ix, iy]
                    DBF[ii]   += min(np.abs(ix-iy), len(x)-np.abs(ix-iy))*Corr["NbNf"][ix, iy]
                else:
                    DBB[ii]   += np.abs(ix-iy)*Corr["NbNb"][ix, iy]
                    DFF[ii] += np.abs(ix-iy)*Corr["NfNf"][ix, iy]
                    DBF[ii]  += np.abs(ix-iy)*Corr["NbNf"][ix, iy]
        DBB[ii]   = (DBB[ii]   - NB )/(NB*(NB-1))   if NB*(NB-1)   > 0.5 else np.NaN 
        DFF[ii]   = (DFF[ii]   - NF )/(NF*(NF-1))   if NF*(NF-1)   > 0.5 else np.NaN 
        DBF[ii]   = (DBF[ii]        )/(NB*NF)       if NB*NF       > 0.5 else np.NaN 

    DBB  = np.array(DBB)
    DFF  = np.array(DFF)
    DBF  = np.array(DBF)

    Distance = {"DBB": DBB,
                "DFF": DFF,
                "DBF": DBF}

    return Param, Distance

def VonNeumannEntropy(A, ON=False):
    rho = np.flip(np.linalg.eigvalsh(A))
    N = sum(rho)
    S = 0
    for ii in range(len(rho)):
        if rho[ii] > 0.0:
            S += -rho[ii]*np.log(rho[ii])

    if N > 0.0:
        S = S/N+np.log(N)

    if ON:
        return S, rho
    else:
        return S


def get_OccupationNumber(folder, parameter):
    filepath = glob.glob(folder+"*.cout")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-5].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                param = float(x[n:])
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]
    
    ONb = []
    Sb  = []
    ONf = []
    Sf  = []
    
    for fp in filepath:
        Corr = get_Correlation(fp)
        sb, onb = VonNeumannEntropy(Corr["BtB"], True)
        sf, onf = VonNeumannEntropy(Corr["CtC"], True)
        ONb.append(onb)
        ONf.append(onf)
        Sb.append(sb)
        Sf.append(sf)
    
    ONb = np.array(ONb)
    ONf = np.array(ONf)
    Sb = np.array(Sb)
    Sf = np.array(Sf)

    OccNum = {"ONb": ONb,
              "ONf": ONf,
              "Sb": Sb,
              "Sf": Sf}

    return Param, OccNum

def get_NaturalOrbitals(folder, parameter, nb=1, nf=1):
    filepath = glob.glob(folder+"*.cout")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-5].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                param = float(x[n:])
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]

    NOb = []
    for ii in range(nb):
        NOb.append([])
    NOf = []
    for ii in range(nf):
        NOf.append([])
    
    for fp in filepath:
        Corr = get_Correlation(fp)
        eigvals_b, eigvecs_b = np.linalg.eigh(Corr["BtB"])
        for ii in range(nb):
            vb = eigvecs_b[:, -ii-1]
            NOb[ii].append(np.abs(vb)*np.abs(vb)/np.linalg.norm(vb)**2)
        eigvals_f, eigvecs_f = np.linalg.eigh(Corr["CtC"])
        for ii in range(nf):
            vf = eigvecs_f[:, -ii-1]
            NOf[ii].append(np.abs(vf)*np.abs(vf)/np.linalg.norm(vf)**2)

    for ii in range(nb):
        NOb[ii] = np.array(NOb[ii])
    for ii in range(nf):
        NOf[ii] = np.array(NOf[ii])

    NatOrb = {"NOb": NOb,
              "NOf": NOf}

    return Param, NatOrb

def get_Energies(folder, parameter, periodic=False, UBB=0., UBF=0.):
    filepath = glob.glob(folder+"*.cout")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-5].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                param = float(x[n:])
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]
    
    # Missing terms for general model
    Hopping_B  = [] 
    Hopping_FU = []
    Hopping_FD = []
    BoseBose   = []
    BoseFermi  = []

    for (ii, fp) in enumerate(filepath):
        Corr = get_Correlation(fp)

        BtB = np.diag(Corr["BtB"], 1)
        if periodic:
            np.append(BtB, Corr["BtB"][0,-1])
        CtC   = np.diag(Corr["CtC"], 1)
        Nb = np.diag(Corr["BtB"])
        NbNb = np.diag(Corr["NbNb"])
        NbNf = np.diag(Corr["NbNf"])

        Hopping_B.append(-2*sum(BtB))
        Hopping_F.append(-2*sum(CtC))
        if parameter == "UBB":
            BoseBose.append(Param[ii]*sum(NbNb-Nb)/2)
        else:
            BoseBose.append(UBB*sum(NbNb-Nb)/2)
        if parameter == "UBF":
            BoseFermi.append(Param[ii]*sum(NbNf))
        else:
            BoseFermi.append(UBF*sum(NbNf))

    Hopping_B = np.array(Hopping_B)
    Hopping_F = np.array(Hopping_F)
    BoseBose   = np.array(BoseBose)
    BoseFermi  = np.array(BoseFermi)

    Energies = {"Hopping_B": Hopping_B,
                "Hopping_F": Hopping_F,
                "BoseBose": BoseBose,
                "BoseFermi": BoseFermi}

    return Param, Energies

def get_TwoBodyFermion(folder, parameter):
    filepath = glob.glob(folder+"*.ccout")
    file = [fp.split("/")[-1] for fp in filepath]
    
    Param = np.zeros(len(file))
    for (ii, f) in enumerate(file):
        fs = f[:-6].split("_")
        param = 0.
        for x in fs:
            n = len(parameter)
            if x[0:n] == parameter:
                param = float(x[n:])
        Param[ii] = param
    
    sort_order = np.argsort(Param)
    Param = np.array(Param)[sort_order]
    file = np.array(file)[sort_order]
    filepath = np.array(filepath)[sort_order]
    
    CtCtCC = []

    for (ii, fp) in enumerate(filepath):
        CtCtCC_ii = np.genfromtxt(fp)

        #reorder?
        CtCtCC.append(CtCtCC_ii)

    return Param, CtCtCC

def get_ExactDiagonalization(Vb):
    eigval, eigvec = eigh_tridiagonal(Vb, -np.ones(len(Vb)-1))
    return eigval, eigvec.T

def get_DensityExact(Nf, eigvec):
    n = np.zeros(len(eigvec[0]))
    for ii in range(Nf):
        n += eigvec[ii]**2
    return n   

def get_TwoBodyDensity(ii, jj, eigvec):
    return np.outer(eigvec[ii], eigvec[jj])-np.outer(eigvec[jj], eigvec[ii])

def get_TwoBodyOrbitals(eigval, eigvec):
    TwoBodyEnergy = []
    TwoBodyOrbitals = []
    for ii in range(len(eigval)):
        for jj in range(ii+1, len(eigval)):
            TwoBodyEnergy.append(eigval[ii]+eigval[jj])
            TwoBodyOrbitals.append(get_TwoBodyDensity(ii, jj, eigvec))

    sort_order      = np.argsort(TwoBodyEnergy)
    TwoBodyEnergy   = np.array(TwoBodyEnergy)[sort_order]
    TwoBodyOrbitals = np.array(TwoBodyOrbitals)[sort_order]
    return TwoBodyEnergy, TwoBodyOrbitals

def FermionSymmetrization(v):
    N = len(v)
    M = 0.5*(np.sqrt(1+8*N)+1)
    M = int(M)
    A = np.zeros((M,M))

    kk = 0
    for ii in range(M):
        for jj in range(ii+1, M):
            A[ii, jj] = v[kk]**2
            kk += 1

    return A + A.T
