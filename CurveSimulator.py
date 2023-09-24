import os
import sys
import numpy as np

from math import gcd
from numpy import pi, sin, tan
import multiprocessing as mp
import concurrent.futures as fut
from plot import MakeDir, MultiRankHistPos, MultiXYPlot

import ctypes

libDS_path = os.path.join("CPP/cmake-build-release", 'libDS.so')
sys.path.append("CPP/cmake-build-release")
if sys.platform == 'linux':
    libDS = ctypes.cdll.LoadLibrary(libDS_path)
c_double_p = ctypes.POINTER(ctypes.c_double)
c_int64_p = ctypes.POINTER(ctypes.c_int64)
c_uint64_p = ctypes.POINTER(ctypes.c_uint64)
c_size_t_p = ctypes.POINTER(ctypes.c_size_t)

class c_double_complex(ctypes.Structure):
    """complex is a c structure
    https://docs.python.org/3/library/ctypes.html#module-ctypes suggests
    to use ctypes.Structure to pass structures (and, therefore, complex)
    """
    _fields_ = [("real", ctypes.c_double),("imag", ctypes.c_double)]

    @property
    def value(self):
        return self.real+1j*self.imag # fields declared above
c_double_complex_p = ctypes.POINTER(c_double_complex)


def DS_CPP(n, m, N_pos, N_neg, beta):
    INT64 = ctypes.c_int64
    libDS.DS.argtypes = (INT64, INT64, INT64, INT64, ctypes.c_double, c_double_p)
    libDS.DS.restype = ctypes.c_double
    np_o_o = np.zeros(1, dtype=float)
    dsabs = libDS.DS(n, m, N_pos, N_neg, beta, np_o_o.ctypes.data_as(c_double_p))
    return dsabs, np_o_o[0]


def SPECTRUM_CPP( N_pos, N_neg,N_lam, beta, gamma,tol):
    #  void FindSpectrumFromResolvent(std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_lam,double beta, std::complex<double> gamma, 
    #/*OUT*/std::complex<double> * lambdas, bool cold_start, double tol);
    N = N_pos + N_neg
    lambdas = np.zeros(N_lam,dtype= complex)
    lambdas.fill( 1j)
    # lambdas.fill(0 + 0j)
    INT64 = ctypes.c_int64
    func = libDS.FindSpectrumFromResolvent
    func.argtypes = (INT64, INT64, INT64, ctypes.c_double, c_double_complex,c_double_complex_p ,ctypes.c_bool,ctypes.c_double)
    func.restype = ctypes.c_uint64
    arg_gamma = c_double_complex(gamma, 0)
    N_good = func(N_pos, N_neg,N_lam, beta, arg_gamma,
                                    lambdas.ctypes.data_as(c_double_complex_p),False,tol)
    
    return np.sort_complex(lambdas[:N_good]) if N_good > 0 else None


def CorrFuncDir(Mu):
    return os.path.join("plots", "VorticityCorr." + str(Mu))

class CurveSimulator():
    def __init__(self, Mu, EG, T, CPU, R0, R1, STP, C, Nlam, gamma):
        MakeDir(CorrFuncDir(Mu))
        self.mu = Mu
        self.CPU = CPU
        self.C = C
        self.T = T
        self.EG = EG
        self.R0 = R0
        self.R1 = R1
        self.STP = STP
        self.spectralParams =(Nlam, gamma,1e-4)
        self.lambdas = np.zeros(Nlam, dtype=complex)
        self.M = 16

    def EulerPair(self):
        if self.M ==0:
            M = int(np.clip(np.floor(0.5*np.random.exponential(1./self.mu)),3,5e9))*2
        else:
            M = self.M

        while (True):
            q = 2 * np.random.randint(2,M//2)
            p = np.random.randint(1, q)
            if gcd(p,q) ==1:  # P( (p,q)=1 ) = phi(q)/q
                if np.random.randint(0,M) < q:  # P( break ) = q/M * phi(q)/q = phi(q)/M ~ phi(q)
                    break
                pass
            pass
        return [M, p, q]

        
    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end - beg) * 3, dtype=float).reshape(-1, 3)
        np.random.seed(self.C + 1000 * beg)  # to make a unique seed for at least 1000 nodes

        for k in range(beg, end):
            M, p, q = self.EulerPair()
            beta = (2 * pi * p) / float(q)
            r =0
            N_pos = (M + q*r) // 2  # Number of 1's
            N_neg = M - N_pos  # Number of -1's

            n = np.random.randint(0, M)
            m = np.random.randint(n + 1, M + n) % M
            if n > m:
                n, m = m, n

            t = k - beg
            ar[t, 0] = 1/tan(beta/2)**2
            ar[t, 1], ar[t, 2] = DS_CPP(n, m, N_pos, N_neg, beta)
        return ar
    
    def getSpectrum(self, params):
        beg, end = params
        N_lam, gamma, tol = self.spectralParams
        ar = np.zeros(shape=(end-beg,N_lam),dtype = complex)
        ar.fill(np.NaN)
        np.random.seed(self.C + 1000 * beg)  # to make a unique seed for at least 1000 nodes

        for k in range(beg, end):
            M, p, q = self.EulerPair()
            beta = (2 * pi * p) / float(q)
            r =0
            N_pos = (M + q*r) // 2  # Number of 1's
            N_neg = M - N_pos  # Number of -1's
            t = k - beg
            res  = SPECTRUM_CPP( N_pos, N_neg,N_lam, beta, gamma,tol)
            if res is None: continue
            if len(res) >0:
                ar[t,:len(res)] = res[:]
            pass
        return ar
    
    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.mu), "Fdata." + str(self.EG)+ "."+ str(self.T) + "." + str(self.C) + ".np")
    
    def SpectrumPathname(self):
        return os.path.join(CorrFuncDir(self.mu), "Spectrum."+ str(self.spectralParams) + "." + str(self.C) + ".np")

    def FDistribution(self, serial):
        """
        :param serial: Boolean If set, run jobs serially.
        """
        mu = self.mu
        T = self.T
        MakeDir(CorrFuncDir(mu))
        if not os.path.isfile(self.FDistributionPathname()):
            res = None
            params = [(T * i // self.CPU, T * (i + 1) // self.CPU) for i in range(self.CPU)]
            if serial:
                res = list(map(self.GetSamples, params))
            else:
                with fut.ProcessPoolExecutor(max_workers=self.CPU) as exec:
                    res = list(exec.map(self.GetSamples, params))
            data = np.vstack(res) #
            data.tofile(self.FDistributionPathname())
        print("made FDistribution " + str(mu))
        
    def PrepareSpectrum(self, serial):
        """
        :param serial: Boolean If set, run jobs serially.
        """
        mu = self.mu
        T = self.T
        Nlam = self.spectralParams[0]
        self.lambdas.fill(0+0j)

        MakeDir(CorrFuncDir(mu))
        os.remove(self.SpectrumPathname())
        # if not os.path.isfile(self.SpectrumPathname()):
        with open(self.SpectrumPathname(), 'a') as fout:
            for iter in range(10):
                res = None
                params = [(T * i // self.CPU, T * (i + 1) // self.CPU, ) for i in range(self.CPU)]
                if serial:
                    res = list(map(self.getSpectrum, params))
                else:
                    with fut.ProcessPoolExecutor(max_workers=self.CPU) as exec:
                        res = list(exec.map(self.getSpectrum, params))
                data = np.vstack(res).reshape(Nlam,-1)
                self.lambdas = np.zeros(Nlam,complex)
                for n in range(Nlam):
                    ok = ~np.isnan(data[n])
                    good_data = data[n][ok]
                    self.lambdas[n] = good_data.mean()
                    xx = np.real(good_data)
                    err = np.std(xx)
                    print(" n=", n, " mean lambda =", self.lambdas[n] , " +/- ",err )
                np.savetxt(fout, [iter, self.M] + self.lambdas)
                self.M *= 2
        print("made Spectrum " + str(mu))
        
    @staticmethod
    def ReadStatsFile(params):
        pathname, T , dim= params
        return np.fromfile(pathname, float).reshape(T, dim)

    
    def GetFDStats(self,mu):
        params= []
        T = None
        for filename in os.listdir(CorrFuncDir(mu)):
            if filename.endswith(".np") and filename.startswith("Fdata." + str(self.EG)):
                try:
                    splits = filename.split(".")
                    T = int(splits[-3])
                    if T > 0:
                        params.append([os.path.join(CorrFuncDir(mu), filename), T,3])
                except Exception as ex:
                    pass
                pass
            pass
        res = []
        with fut.ProcessPoolExecutor(max_workers=mp.cpu_count() -1) as exec:
            res = list(exec.map(self.ReadStatsFile, params))
        if res ==[]:
            raise Exception("no stats to collect!!!!")
        data = np.vstack(res).reshape(-1, T, 3)
        pathname = os.path.join(CorrFuncDir(self.mu), 'FDStats.' + str(self.EG)+ "." + str(T) + '.np')
        data.tofile(pathname)
        return pathname, T
        
    def MakePlots(self, Mulist):
        Betas =[]
        Dss = []
        OdotO = []
        np.sort(Mulist)
        for mu in Mulist:
            pathname = None
            T = None
            for filename in os.listdir(CorrFuncDir(mu)):
                if filename.endswith(".np") and filename.startswith("FDStats." + str(self.EG)):
                    try:
                        splits = filename.split(".")
                        T = int(splits[-2])
                        if (T > 0): 
                            pathname = os.path.join(CorrFuncDir(self.mu), filename)
                            break
                    except Exception as ex:
                        break
                    pass
            if pathname is None:
                pathname, T = self.GetFDStats(mu)
            array = np.fromfile(pathname,dtype=float)
            stats = array.reshape(-1, T, 3)
            stats = np.transpose(stats, (2, 1, 0)).reshape(3,-1)
            Betas.append(stats[0])
            Dss.append(stats[1])
            OdotO.append(stats[2])
        pass
        MaxMu = Mulist[-1]
        if sys.platform == 'linux':
            for X, name in zip([Betas, Dss, OdotO, OdotO], ["logTanbeta", "DS", "OmOm", "-OmOm"]):
                data = []
                for k, mu in enumerate(Mulist):
                    data.append([str(mu), X[k]]) if name != "-OmOm" else data.append([str(mu), -X[k]])
                plotpath = os.path.join(CorrFuncDir(MaxMu), str(self.EG)+ "."  + name + ".png")
                try:
                    logx = True
                    logy = True
                    MultiRankHistPos(data, plotpath, name, logx=logx, logy=logy, num_subsamples=1000)
                except Exception as ex:
                    print(ex)
            pass
            data = []
            for k, mu in enumerate(Mulist):
                oto = OdotO[k]
                dss =Dss[k]
                pos = oto > 0
                neg = oto < 0
                data.append([str(mu), dss[pos], oto[pos]])
                data.append([str(-mu), dss[neg], -oto[neg]])
                try:
                    plotpath = os.path.join(CorrFuncDir(MaxMu), str(self.EG)+ ".OtOvsDss.png")
                    MultiXYPlot(data, plotpath, logx=True, logy=True, title='OtoOVsDss', scatter=False, xlabel='log(dss)',
                                ylabel='log(oto)', frac_last=0.9, num_subsamples=1000)
                except Exception as ex:
                    print(ex)
            print("plotted otovsds " + str(MaxMu))

        if self.STP<=0: return
        data =[]
        rho_data = np.linspace(self.R0,self.R1,self.STP)
        for k, mu in enumerate(Mulist):
            oto = OdotO[k]
            dss =Dss[k]
            corr = np.zeros_like(rho_data)
            rdx = rho_data[1:,np.newaxis]* dss[np.newaxis,:]
            otX = oto[np.newaxis,:]
            corr[1:] = np.mean(otX * sin(rdx)/rdx,axis=1)
            corr[0] = np.mean(oto)
            data.append([str(mu), rho_data, corr])
        try:
            plotpath = os.path.join(CorrFuncDir(MaxMu), str(self.EG)+ ".CorrFunction.png")
            MultiXYPlot(data, plotpath, logx=True, logy=True, title='CorrFunction', scatter=False, xlabel='rho',
                        ylabel='cor', frac_last=0.9, num_subsamples=1000)
        except Exception as ex:
            print(ex)
        print("plotted CorrFunction " + str(MaxMu))
            
