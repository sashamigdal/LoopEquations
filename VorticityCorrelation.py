
import os

from mpmath import rational

from SortedArrayIter import SortedArrayIter
from Timer import MTimer as Timer
from plot import MakeDir, XYPlot, SubSampleWithErr, PlotXYErrBars, MultiPlot, MultiRankHistPos, MultiXYPlot
# from functools import reduce
# from operator import add

from numpy import pi, sin, cos, tan, sqrt, exp, log
import numpy as np
from parallel import ConstSharedArray
import multiprocessing as mp
import concurrent.futures as fut
from fractions import Fraction
def CorrFuncDir(M):
    return os.path.join("plots", "VorticityCorr." + str(M))


def F(k, alphas, beta):
    return 1 / (2 * sin(beta / 2)) * np.array([cos(alphas[k]), sin(alphas[k]), 1j * cos(beta / 2)], complex)


def Omega(k, alphas, beta):
    return (alphas[k + 1] - alphas[k]) / (2 * beta * tan(beta / 2)) * np.array(
        [cos((alphas[k] + alphas[k + 1]) / 2), sin((alphas[k] + alphas[k + 1]) / 2), 1j], complex)

class RandomFractions():
    @staticmethod
    def Pair( fM):
        f,M = fM
        f = np.clip(f,0.5/M,1-0.5/M)
        pq = Fraction(f).limit_denominator(M//2)
        eps = M%2
        p,q =  [pq.numerator,2*pq.denominator+ eps]
        return [p,q]

    def __init__(self,M, T):
        self.M = M
        self.T = T
        MakeDir(self.SaveDir())

    def SaveDir(self):
        return os.path.join("plots", "RandomFractions")

    def GetPathname(self):
        return os.path.join(self.SaveDir(), str(self.M)+ ".np")

    def MakePairs(self):
        T = self.T
        if os.path.isfile(self.GetPathname()):
            pairs = np.fromfile(self.GetPathname(),dtype=int).reshape(-1,2)
            if len(pairs) >= T:
                return pairs[:T]
            pass
        ff = list(zip(np.random.random(size=T),[self.M]*T))
        res = []
        with fut.ProcessPoolExecutor() as exec:
            res = list(exec.map(self.Pair, ff))
        pairs = np.array(res,dtype= int)
        pairs = np.unique(pairs,axis=0)
        np.random.shuffle(pairs)
        pairs = pairs[:self.T]
        pairs.tofile(self.GetPathname())
        return pairs


class FDPlotter():
    def GetSamples(self, params):
        beg, end = params
        ar = np.zeros((end-beg)*3,dtype=float).reshape(-1,3)
        for k in range(beg, end):
            i = self.Mindex(k)
            M = (i+1)*self.M
            p,q = list(self.pq[i][k])
            beta = (2 * pi * p) / float(q)
            # if np.random.randint(2) ==1 :
            #     beta =-beta
            N1,N2 = (M+q)//2,(M-q)//2
            if np.random.randint(2) ==1 :
                N1,N2 = N2,N1
            alphas = np.array([1] * N1 + [-1] *N2 , dtype=int)
            np.random.shuffle(alphas)
            alphas = np.cumsum(alphas)
            alphas = np.append(alphas, alphas[0]).astype(float) * beta

            FF = np.vstack([F(k, alphas, beta) for k in range(M)])
            FS = np.cumsum(FF, axis=0)
            Stot = FS[-1]
            m = np.random.randint(1, M)
            n = np.random.randint(0, m)
            Snm = FS[m] - FS[n]
            Smn = Stot - Snm
            snm = Snm / (m - n)
            smn = Smn / (n + M - m)
            ds = snm.real - smn.real
            t = k-beg
            ar[t,0] = beta
            ar[t,1] = sqrt(ds.dot(ds))
            ar[t,2] = np.dot(Omega(n, alphas, beta), Omega(m, alphas, beta)).real
        return ar

    def GetCorr(self, params):
        i,j, beg, end  = params
        s = self.dss[i:j, np.newaxis] * self.Rdata[np.newaxis, beg:end]
        s = sin(s) / s
        return self.OdotO[i:j].dot(s)/(j-i)

    def PlotWithErr(self, xdata, ydata, name, num_samples=100):
        x, y, yerr =SubSampleWithErr(xdata,ydata, num_samples )
        plotpath = os.path.join(CorrFuncDir(self.M),name + ".png")
        PlotXYErrBars(x, y, yerr, plotpath,title=name)

    def FDistributionPathname(self):
        return os.path.join(CorrFuncDir(self.M), "Fdata." + str(self.T) + ".np")

    def CorrDataPathname(self):
        return os.path.join(CorrFuncDir(self.M), "CorrData." + str(self.T) + ".np")
    def MDataPathname(self):
        return os.path.join(CorrFuncDir(self.M), "MData." + str(self.T) + ".np")

    def Mindex(self, k):
        return (4 * k)//self.T


    def FDistribution(self):
        M = self.M
        T = self.T
        R = self.R
        Tstep = self.Tstep

        MakeDir(CorrFuncDir(M))
        try:
            data = np.fromfile(self.FDistributionPathname(),dtype=float).reshape(-1,3)
        except:
            res = []
            params = [[k, k + Tstep ] for k in range(0, T, Tstep)]
            with fut.ProcessPoolExecutor() as exec:
                res = list(exec.map(self.GetSamples, params))
            data = np.vstack(res)
            data.tofile(self.FDistributionPathname())
            print("made FDistribution " + str(M))
        data = data.T
        self.betas = ConstSharedArray(data[0])
        self.dss = ConstSharedArray(data[1])
        self.OdotO = ConstSharedArray(data[2])
        self.Rdata = ConstSharedArray(np.linspace(1./M, 0.5, R))


    def MakeCorrData(self):
        M = self.M
        R = self.R
        T = self.T
        Rstep = self.Rstep
        if os.path.isfile(self.CorrDataPathname()):
            print("already made parallel map GetCorr " + str(M))
            return

        dd =[]
        for i,j, m in [(T*k//4,T*(k+1)//4,(k+1)*M) for k in range(4)]:
            res = []
            with fut.ProcessPoolExecutor() as exec:
                ranges = [[ i,j, k, k + Rstep] for k in range(0, R, Rstep)]
                res.extend(list(exec.map(self.GetCorr, ranges)))
            dd.extend([np.hstack(res)])
        corrdata = np.stack(dd)
        corrdata.tofile(self.CorrDataPathname())
        print("made parallel map GetCorr " + str(M))
        corrdata = np.fromfile(self.CorrDataPathname(),dtype=float).reshape(4,-1)
        datapos = []
        dataneg= []
        for i in range(4):
            m = (1+1)*self.M
            pos = corrdata[i] >0
            neg = corrdata[i] <0
            datapos.append([str(m),self.Rdata[pos],corrdata[i][pos]])
            dataneg.append([str(m),self.Rdata[neg],-corrdata[i][neg]])
        plotpath = os.path.join(CorrFuncDir(M),"multicorrPos.png")
        try:
            MultiXYPlot(datapos, plotpath, logx=True, logy=True, title='VortCorrPos',scatter=False, xlabel ='r', ylabel='corr')
        except Exception as ex:
            print(ex)
        plotpath = os.path.join(CorrFuncDir(M),"multicorrNeg.png")
        try:
            MultiXYPlot(dataneg, plotpath, logx=True, logy=True, title='VortCorrNeg',scatter=False, xlabel ='r', ylabel='-corr')
        except Exception as ex:
            print(ex)
        print("made VortCor " + str(M))


    def MakeOmtoDSFit(self):
        data = []
        M = self.M
        T = self.T
        for i,j, m in [(T*k//4,T*(k+1)//4,(k+1)*M) for k in range(4)]:
            oto = self.OdotO[i:j]
            pos = oto >0
            dss = self.dss[i:j]
            data.append([str(m),dss[pos],oto[pos]])
        try:
            plotpath = os.path.join(CorrFuncDir(M),"multi OtOvsDss.png")
            MultiXYPlot(data, plotpath, logx=True, logy=True, title='OtoOVsDss',scatter=False, xlabel ='dss', ylabel='oto',frac_last=0.5)
        except Exception as ex:
            print(ex)
        print("made OtOvsDss " + str(M))

    def __init__(self, M, T, R):
        MakeDir(CorrFuncDir(M))
        self.M = M
        self.Tstep = int(T / (mp.cpu_count() - 1)) + 1
        T = self.Tstep * (T//self.Tstep)
        self.T = T
        self.Rstep =  int(R / (mp.cpu_count() - 1)) + 1
        R =  self.Rstep * (R//self.Rstep)
        self.R = R
        self.pq = [None]*4
        with Timer("making pairs "):
            for i in range(4):
                self.pq[i] =ConstSharedArray(RandomFractions((i+1)*M,T).MakePairs())
            pass
        self.FDistribution()
        for x, name in zip([self.betas,self.dss,-self.OdotO[:], self.OdotO[:]],["beta","DS","-OmOm","OmOm"]):
            data = []
            for beg,end, m in [(T*k//4,T*(k+1)//4,(k+1)*M) for k in range(4)]:
                data.append([str(m),x[beg:end]])
            plotpath = os.path.join(CorrFuncDir(M),"multi " + name + ".png")
            try:
                logx = (name in ("DS", "OmOm", "-OmOm"))
                logy = (name != "beta")
                MultiRankHistPos(data,plotpath,name,logx=logx,logy=logy)
            except Exception as ex:
                print(ex)
        pass
        self.MakeOmtoDSFit()


def test_FDistribution():
    M = 10001
    T = 1000
    R = 1000
    # pq= RandomFractions(M,T).MakePairs()
    with Timer("done FDistribution for M,T,R= " + str(M) + "," + str(T) + "," + str(R)):
        fdp = FDPlotter(M, T, R)

if __name__ == '__main__':
    test_FDistribution()
