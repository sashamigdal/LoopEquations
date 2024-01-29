import os
import sys
import numpy as np
import numpy.typing as npt

from math import gcd
from numpy import pi, sin, tan
import multiprocessing as mp
import concurrent.futures as fut
from plot import MakeDir, MultiRankHistPos, MultiXYPlot
import ctypes
from Timer import GetTime
from time import sleep


lib_config = "debug" if (os.getenv("USE_DEBUG_LIB") == "1") else "release"
cxx_lib_dir = "CPP/cmake-build-" + lib_config
if sys.platform == "linux":
    cxx_lib_ext = "so"
elif sys.platform == "win32":
    cxx_lib_ext = "dll"
libDS_path = os.path.join(cxx_lib_dir, 'libEuler.' + cxx_lib_ext)
libEulerGPU_path = os.path.join(cxx_lib_dir, 'libEulerGPU.' + cxx_lib_ext)
sys.path.append(cxx_lib_dir)

libDS = None
libEulerGPU = None

c_int64 = ctypes.c_int64
c_uint64 = ctypes.c_uint64
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


def DS_CPP(n, m, N_pos, N_neg, q, beta):
    libDS.DS.argtypes = (c_int64, c_int64, c_int64, c_int64, c_int64, ctypes.c_double, c_double_p)
    libDS.DS.restype = ctypes.c_double
    np_o_o = np.zeros(1, dtype=float)
    dsabs = libDS.DS(n, m, N_pos, N_neg, q, beta, np_o_o.ctypes.data_as(c_double_p))
    return dsabs, np_o_o[0]

def DS_GPU(warp_size : int,
           ns : npt.NDArray[np.int64],
           ms : npt.NDArray[np.int64],
           N_poss : npt.NDArray[np.int64],
           N_negs : npt.NDArray[np.int64],
           qq : npt.NDArray[np.int64],
           betas : npt.NDArray[np.double]) -> (npt.NDArray[np.double], npt.NDArray[np.double]):
    """
    Wrapper on C++ CUDA code for GPU
    :param warp_size: Warp size on the GPU
    :param ns: array of n's
    :param ms: array of m's
    :param N_poss: array of N_pos's
    :param N_negs: array of N_neg's
    :param q: array of q's
    :param betas: array of beta's
    :return: pair of arrays
    """
    nSamples = len(ns)
    Ss = np.zeros(nSamples * warp_size, dtype=np.double)
    o_os = np.zeros(nSamples * warp_size, dtype=np.double)
    libEulerGPU.DS_GPU.argtypes = (c_uint64, c_uint64_p, c_uint64_p, c_uint64_p, c_uint64_p, c_uint64_p, c_double_p, c_double_p)
    libEulerGPU.DS_GPU(nSamples, ns.ctypes.data_as(c_uint64_p), ms.ctypes.data_as(c_uint64_p),
                 N_poss.ctypes.data_as(c_uint64_p), N_negs.ctypes.data_as(c_uint64_p), qq.ctypes.data_as(c_uint64_p),
                       betas.ctypes.data_as(c_double_p),
                 Ss.ctypes.data_as(c_double_p), o_os.ctypes.data_as(c_double_p) )
    return Ss, o_os

def DS_GetGpuWarpSize():
    libEulerGPU.GetGpuWarpSize.restype = ctypes.c_int
    return libEulerGPU.GetGpuWarpSize()

def SPECTRUM_CPP( N_pos, N_neg,N_lam, beta, gamma,tol):
    #  void FindSpectrumFromResolvent(std::int64_t N_pos, std::int64_t N_neg, std::int64_t N_lam,double beta, std::complex<double> gamma, 
    #/*OUT*/std::complex<double> * lambdas, bool cold_start, double tol);
    N = N_pos + N_neg
    lambdas = np.zeros(N_lam,dtype= complex)
    lambdas.fill( 1j)
    # lambdas.fill(0 + 0j)
    INT64 = ctypes.c_int64
    UINT64 = ctypes.c_uint64

    bUseResolvent = True
    func = libDS.FindSpectrumFromResolvent if bUseResolvent else libDS.FindSpectrumFromSparsematrix
    func.argtypes = (INT64, INT64, UINT64, ctypes.c_double, c_double_complex,c_double_complex_p, ctypes.c_bool,ctypes.c_double)
    func.restype = ctypes.c_uint64

    arg_gamma = c_double_complex(gamma, 0)
    N_good = func(N_pos, N_neg, N_lam, beta, arg_gamma,
                                    lambdas.ctypes.data_as(c_double_complex_p),False,tol)
    
    return np.sort_complex(lambdas[:N_good]) if N_good > 0 else None


def CorrFuncDir(M : int, compute : str, run : int) -> str:
    return os.path.join("plots", f"VorticityCorr.{M}.{compute}.{run}")

class CurveSimulatorBase():
    def __init__(self, M, EG, T, CPU, R0, R1, STP, run, C, compute):
        self.M = M
        self.nWorkers = CPU
        self.C = C
        self.T = T
        self.EG = EG
        self.R0 = R0
        self.R1 = R1
        self.STP = STP
        self.run = run
        self.compute = compute
        MakeDir(self.CurrentCorrFuncDir())

    def CurrentCorrFuncDir(self) -> str:
        return CorrFuncDir(self.M, self.compute, self.run)

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

    def EulerPairWithLogCot(self):
        if self.M ==0:
            M = int(np.clip(np.floor(0.5*np.random.exponential(1./self.mu)),3,5e9))*2
        else:
            M = self.M

        while (True):
            #tested in Mathematica (TestCotCDF.nb)
            q = 2 * np.random.randint(2,M//2)
            r = np.log(np.tan(pi/q))
            p = int(np.floor(q/pi * np.arctan(np.exp(r * (2 * np.random.random() - 1))/q)))
            if np.random.random() < 0.5 :
                p = q - p
            if gcd(p,q) ==1:  # P( (p,q)=1 ) = phi(q)/q
                return [M, p, q]
        return None
class CurveSimulatorFDistribution(CurveSimulatorBase):
    def __init__(self, M, EG, T, CPU, run, C, compute):
        CurveSimulatorBase.__init__(self, M, EG, T, CPU, 0, 0, 0, run, C, compute)

    def DoWork(self, serial):
        """
        :param serial: Boolean If set, run jobs serially.
        """
        global libDS
        global libEulerGPU
        T = self.T
        MakeDir(self.CurrentCorrFuncDir())

        if not os.path.isfile(self.Pathname()):
            res = None
            t0 = GetTime()
            if self.compute == "CPU":
                if sys.platform == 'linux' or sys.platform == "win32":
                    libDS = ctypes.cdll.LoadLibrary(libDS_path)
                params = range(self.nWorkers)
                if serial:
                    res = list(map(self.GetSamples, params))
                else:
                    with fut.ProcessPoolExecutor(max_workers=self.nWorkers) as exec:
                        res = list(exec.map(self.GetSamples, params))
                Mtot = T * self.M
            elif self.compute == "GPU":
                if sys.platform == 'linux' or sys.platform == "win32":
                    libEulerGPU = ctypes.cdll.LoadLibrary(libEulerGPU_path)
                res = [self.GetSamples(0)]
                Mtot = T * self.M * 32
            dt = GetTime() - t0
            data = np.vstack(res)
            data.tofile(self.Pathname())
            print(f"made FDistribution {self.M}")
            speed = Mtot / dt
            print(f"Random walker has made {Mtot} steps total in {dt} seconds." +
                  f" Avg speed is {speed:g} steps per second.")

    def Pathname(self):
        return os.path.join(self.CurrentCorrFuncDir(), "Fdata." + str(self.EG)+ "."+ str(self.T) + "." + str(self.C) + ".np")

    def GenerateEulerSet(self):
        M, p, q = self.EulerPairWithLogCot()
        beta = (2 * pi * p) / float(q)
        r = 0
        N_pos = (M + q * r) // 2  # Number of 1's
        N_neg = M - N_pos  # Number of -1's

        n = np.random.randint(0, M)
        m = np.random.randint(n + 1, M + n) % M
        if n > m:
            n, m = m, n
        return n, m, N_pos, N_neg, beta, q

    def GetSamples(self, workerId):
        partition = 0 if self.compute == "CPU" else 1
        # to make a unique seed for at least 1000 nodes
        np.random.seed(((partition * 100 + self.run) * 1000 + self.C) * self.nWorkers + workerId)
        ncols = 4
        if self.compute == "CPU":
            beg = self.T * workerId // self.nWorkers
            end = self.T * (workerId + 1) // self.nWorkers
            ar = np.zeros((end - beg) * ncols, dtype=float).reshape(-1, ncols)
            for k in range(beg, end):
                n, m, N_pos, N_neg, beta, q = self.GenerateEulerSet()
                t = k - beg
                ar[t, 0] = 1 / (q *tan(beta / 2)) ** 2
                ar[t, 1], ar[t, 2] = DS_CPP(n, m, N_pos, N_neg, q, beta)
                ar[t, 3] = q
        elif self.compute == "GPU":
            warp_size = DS_GetGpuWarpSize()
            beg = 0
            end = self.T
            ar = np.zeros((end - beg) * warp_size * ncols, dtype=float).reshape(-1, ncols)
            ns = np.zeros(end - beg, dtype=np.int64)
            ms = np.zeros(end - beg, dtype=np.int64)
            N_poss = np.zeros(end - beg, dtype=np.int64)
            N_negs = np.zeros(end - beg, dtype=np.int64)
            betas = np.zeros(end - beg, dtype=np.double)
            qq = np.zeros(end - beg, dtype=np.int64)
            for k in range(end - beg):
                ns[k], ms[k], N_poss[k], N_negs[k], betas[k], qq[k] = self.GenerateEulerSet()
                ar[k * warp_size : (k + 1) * warp_size, 0] = 1 / (qq[k]*tan(betas[k] / 2) ) ** 2
                ar[k * warp_size : (k + 1) * warp_size, 3] = qq[k]
            Ss, o_os = DS_GPU(warp_size, ns, ms, N_poss, N_negs, qq, betas)
            ar[:, 1] = Ss
            ar[:, 2] = o_os
        return ar

    @staticmethod
    def ReadStatsFile(params):
        pathname, T, dim = params
        return np.fromfile(pathname, float).reshape(T, dim)

    def GetFDStats(self, M):
        params = []
        T = None
        for filename in os.listdir(CorrFuncDir(M, self.compute, self.run)):
            if filename.endswith(".np") and filename.startswith("Fdata." + str(self.EG)):
                try:
                    splits = filename.split(".")
                    T = int(splits[-3])
                    if T > 0:
                        params.append([os.path.join(CorrFuncDir(M, self.compute, self.run), filename), T, 3])
                except Exception as ex:
                    pass
                pass
            pass
        res = []
        with fut.ProcessPoolExecutor(max_workers=mp.cpu_count() - 1) as exec:
            res = list(exec.map(self.ReadStatsFile, params))
        if res == []:
            raise Exception("no stats to collect!!!!")
        data = np.vstack(res).reshape(-1, T, 3)
        pathname = os.path.join(self.CurrentCorrFuncDir(), 'FDStats.' + str(self.EG) + "." + str(T) + '.np')
        data.tofile(pathname)
        return pathname, T

    def MakePlots(self, Mlist):
        Betas = []
        Dss = []
        OdotO = []
        np.sort(Mlist)
        for M in Mlist:
            pathname = None
            T = None
            for filename in os.listdir(CorrFuncDir(M, self.compute, self.run)):
                if filename.endswith(".np") and filename.startswith("FDStats." + str(self.EG)):
                    try:
                        splits = filename.split(".")
                        T = int(splits[-2])
                        if (T > 0):
                            pathname = os.path.join(self.CurrentCorrFuncDir(), filename)
                            break
                    except Exception as ex:
                        break
                    pass
            if pathname is None:
                pathname, T = self.GetFDStats(M)
            array = np.fromfile(pathname, dtype=float)
            stats = array.reshape(-1, T, 3)
            stats = np.transpose(stats, (2, 1, 0)).reshape(3, -1)
            Betas.append(stats[0])
            Dss.append(stats[1])
            OdotO.append(stats[2])
        pass
        MaxMu = Mlist[-1]
        if sys.platform == 'linux':
            for X, name in zip([Betas, Dss, OdotO, OdotO], ["logTanbeta", "DS", "OmOm", "-OmOm"]):
                data = []
                for k, mu in enumerate(Mlist):
                    data.append([str(mu), X[k]]) if name != "-OmOm" else data.append([str(mu), -X[k]])
                plotpath = os.path.join(CorrFuncDir(MaxMu, self.compute, self.run), str(self.EG) + "." + name + ".png")
                try:
                    logx = True
                    logy = True
                    MultiRankHistPos(data, plotpath, name, logx=logx, logy=logy, num_subsamples=1000)
                except Exception as ex:
                    print(ex)
            pass
            data = []
            for k, mu in enumerate(Mlist):
                oto = OdotO[k]
                dss = Dss[k]
                pos = oto > 0
                neg = oto < 0
                data.append([str(mu), dss[pos], oto[pos]])
                data.append([str(-mu), dss[neg], -oto[neg]])
                try:
                    plotpath = os.path.join(CorrFuncDir(MaxMu, self.compute, self.run), str(self.EG) + ".OtOvsDss.png")
                    MultiXYPlot(data, plotpath, logx=True, logy=True, title='OtoOVsDss', scatter=False,
                                xlabel='log(dss)',
                                ylabel='log(oto)', frac_last=0.9, num_subsamples=1000)
                except Exception as ex:
                    print(ex)
            print("plotted otovsds " + str(MaxMu))

        if self.STP <= 0: return
        data = []
        rho_data = np.linspace(self.R0, self.R1, self.STP)
        for k, mu in enumerate(Mlist):
            oto = OdotO[k]
            dss = Dss[k]
            corr = np.zeros_like(rho_data)
            rdx = rho_data[1:, np.newaxis] * dss[np.newaxis, :]
            otX = oto[np.newaxis, :]
            corr[1:] = np.mean(otX * sin(rdx) / rdx, axis=1)
            corr[0] = np.mean(oto)
            data.append([str(mu), rho_data, corr])
        try:
            plotpath = os.path.join(CorrFuncDir(MaxMu, self.compute, self.run), str(self.EG) + ".CorrFunction.png")
            MultiXYPlot(data, plotpath, logx=True, logy=True, title='CorrFunction', scatter=False, xlabel='rho',
                        ylabel='cor', frac_last=0.9, num_subsamples=1000)
        except Exception as ex:
            print(ex)
        print("plotted CorrFunction " + str(MaxMu))


class CurveSimulatorSpectrum(CurveSimulatorBase):
    def __init__(self, M, EG, T, CPU, R0, R1, STP, C, Nlam, gamma):
        CurveSimulatorBase.__init__(M, EG, T, CPU, R0, R1, STP, C)
        self.spectralParams =(Nlam, gamma,1e-4)
        self.lambdas = np.zeros(Nlam, dtype=complex)

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
            res  = SPECTRUM_CPP( N_pos, N_neg, N_lam, beta, gamma,tol)
            if res is None: continue
            if len(res) >0:
                ar[t,:len(res)] = res[:]
            pass
        return ar

    def Pathname(self):
        return os.path.join(CorrFuncDir(), "Spectrum."+ str(self.spectralParams) + "." + str(self.C) + ".np")

    def DoWork(self, M0, serial):
        """
        :param M0: (int) Initial value for `M`, that is doubled a few times.
        :param serial: (boolean) If set, run jobs serially.
        """
        self.M = M0
        M = self.M
        T = self.T
        Nlam = self.spectralParams[0]
        self.lambdas.fill(0+0j)

        MakeDir(CorrFuncDir())
        spectrum_filepath = self.Pathname()
        if os.path.exists(spectrum_filepath):
            os.remove(spectrum_filepath)
        # if not os.path.isfile(self.Pathname()):
        with open(spectrum_filepath, 'a') as fout:
            for iter in range(10):
                res = None
                params = [(T * i // self.nWorkers, T * (i + 1) // self.nWorkers, ) for i in range(self.nWorkers)]
                if serial:
                    res = list(map(self.getSpectrum, params))
                else:
                    with fut.ProcessPoolExecutor(max_workers=self.nWorkers) as exec:
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
        print("made Spectrum " + str(M))
