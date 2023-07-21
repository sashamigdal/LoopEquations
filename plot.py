#!/Users/alexandermigdal/anaconda/bin/python
import shutil
import sys, os
from os import path, system
import numpy as np
import datetime as dt
import time
import os
plotapp = "eog"

def MakeDir(dir):
    try:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    except Exception as e:
        print("makedirs exception :", e)
        pass
    # try:
    #     os.chmod(dir,0777)
    # except Exception as e:
    #     # print "chmod exception :", e
    #     pass

    return dir


def RemDir(dir):
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError:
            print("[ERROR] RemDir({})".format(dir))


def MakeNewDir(dir):
    RemDir(dir)
    try:
        os.makedirs(dir)
    except Exception as e:
        print("makedirs exception :", e)
        pass

def RemoveFiles(dir, word):
    for filename in os.listdir(dir):
        if word in filename:
            path = dir + filename
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
import pickle


TMP_PLOT = '/tmp/plot_dir/'
MakeDir(TMP_PLOT)
import glob

def compress_monthly(times,pnl):
    assert len(times) == len(pnl), "must be parallel"
    N =len(times)
    new_times = []
    new_pnl =[]
    local_time = time.localtime(times[0])
    prev_time = time.mktime(local_time)
    month = local_time.tm_mon

    prev_pnl =0
    for i in range(1,N):
        local_time = time.localtime(times[i])
        if local_time.tm_mon == month and i < N-1: continue
        new_times.append(prev_time)
        p1 = pnl[i-1] if i < N-1 else pnl[i]
        new_pnl.append(p1-prev_pnl)
        prev_time = time.mktime(local_time)
        prev_pnl = pnl[i-1]
        month = local_time.tm_mon


    return new_times,np.array(new_pnl)



def plot_returns(times,pnl,plotpath,label= 'Returns',title="Daily returns"):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dates=map(dt.datetime.fromtimestamp, times)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    pylab.plot(dates,pnl, color="green", linewidth=1., linestyle="-", label=label)
    pylab.legend(loc='upper left')
    plt.title(title)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath

def plot_by_minute_returns(times,pnl,plotpath,label= 'Returns',title="Daily returns"):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dates=map(dt.datetime.fromtimestamp, times)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d.%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    pylab.plot(dates,pnl, color="green", linewidth=1., linestyle="-", label=label)
    pylab.legend(loc='upper left')
    plt.title(title)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath

def plot_monthly_returns(times,pnl,plotpath,relative=True):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    times,pnl = compress_monthly(times,pnl)
    if relative: pnl = 100*(np.exp(pnl)-1)

    dates=np.array(map(dt.datetime.fromtimestamp, times))
    plt.xticks(rotation=25 )
    ax=plt.gca()
    plt.margins(0.05)
    pylab.subplots_adjust(bottom=0.1, right=0.9, top=0.9,left=0.1)
    xfmt = md.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_yticks([0.0], minor=True)
    ax.yaxis.grid()
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.grid()
#    ax.yaxis.grid(True, which='minor')

    pylab.bar(dates[pnl>0],pnl[pnl>0], color="green",   width = 15)
    pylab.bar(dates[pnl<=0],pnl[pnl<=0], color="red",   width = 15)
    #pylab.legend(loc='upper left')
    plt.title("Monthly returns")
    plt.ylabel("Return, \$")
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath


def plot_monthly_return_distrib(times,pnl,plotpath,relative=True):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    times,pnl = compress_monthly(times,pnl)
    if relative: pnl = 100*(np.exp(pnl)-1)

    plt.margins(0.01)
    ax=plt.gca()
    ax.yaxis.grid()
    ax.xaxis.grid()
    plt.hist(pnl,rwidth=0.5,align='mid')
    plt.title("Distribution of monthly returns")
    plt.xlabel("Monthly return, \$")
    plt.ylabel("Number of months")
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath

def plot_histogram(nums,plotpath,title,xlabel="Trades per day", ylabel="Number of days"):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    plt.margins(0.01)
    ax=plt.gca()
    ax.yaxis.grid()
    ax.xaxis.grid()
    plt.hist(nums,rwidth=0.5,align='mid')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath



def plot_daily_return_underwater(times, pnl, plotpath, investment=None):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    if investment is None :
        pnl = np.exp(pnl)
        investment = pnl
    running_max = np.maximum.accumulate(pnl)
    investment_max = np.maximum.accumulate(investment)
    drawdown = (pnl-running_max)/investment_max*100
    # assert (drawdown <=0).all()

    dates=map(dt.datetime.fromtimestamp, times)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    pylab.plot(dates,drawdown, color="blue", linewidth=1., linestyle="-")
    plt.fill_between(dates,0,drawdown,color='b')
    #pylab.legend(loc='upper left')
    plt.ylabel("Drawdown, %")
    plt.title("Underwater Curve")
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath




def DateTest():
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    n=20
    duration=1000
    now=time.mktime(time.localtime())
    timestamps=np.linspace(now,now+duration,n)
    dates=[dt.datetime.fromtimestamp(ts) for ts in timestamps]
    values=np.sin((timestamps-now)/duration*2*np.pi)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates,values)
    plt.show()

def InvestmentPlot(investment, timestamps,plotpath='/tmp/plot.png',title='Investment Plot'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dates=map(dt.datetime.fromtimestamp, timestamps)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    ax.bar(dates,investment,width=1.,color='green')
    plt.ylabel(r'$Investment,$')
    plt.title(title)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath

def PnlImpactPlot(pnl, impact,fee, timestamps,plotpath,title='Pnl Impact Plot',name='Pnl'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dates=map(dt.datetime.fromtimestamp, timestamps)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    pylab.plot(dates,pnl, color="green", linewidth=1., linestyle="-", label=name)
    pylab.plot(dates,impact, color="red", linewidth=1., linestyle="-", label='impact')
    pylab.plot(dates,fee, color="black", linewidth=1., linestyle="-", label='fee')
    plt.ylabel(r'\$')
    pylab.legend(loc='upper left')
    plt.title(title)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath


def DatesPlot(pnl,dates,plotpath,title="Plot"):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter('%Y%m%d')
    ax.xaxis.set_major_formatter(xfmt)
    pylab.plot(dates, pnl, color="green", linewidth=1., linestyle="-", label="pnl")
    plt.ylabel(r'$\ln(p_{T+1}/vwap)$')
    plt.title(title)
    pylab.legend(loc='best')
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath

def PnlDatesPlot(pnl, market, dates,plotpath,title='Performance Plot',name='Plot',market_name='Market',exponentiate=False):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)

    if exponentiate:
        pylab.plot(dates,np.exp(pnl)*1000, color="green", linewidth=1., linestyle="-", label=name)
        pylab.plot(dates,np.exp(market)*1000, color="black", linewidth=1., linestyle="-", label=market_name)
        plt.ylabel(r'Capital, \$')
    else:
        pylab.plot(dates,pnl, color="green", linewidth=1., linestyle="-", label=name)
        pylab.plot(dates,market, color="black", linewidth=1., linestyle="-", label=market_name)
        plt.ylabel(r'$\ln(p_{T+1}/vwap)$')

    plt.title(title)
    pylab.legend(loc='best')
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath


def PnlPlot(pnl, market, timestamps,plotpath,title='Performance Plot',name='Plot',market_name='Market',exponentiate=False):
    dates = map(dt.datetime.fromtimestamp, timestamps)
    return PnlDatesPlot(pnl,market,dates,plotpath,title,name,market_name,exponentiate)

def test_PnlPlot(N=250):
    investment = np.random.uniform(low=0.8,high=1.2,size=N)
    pnl = np.cumsum(np.random.normal(0.001,0.01,size=N))
    market =np.cumsum( np.random.normal(0.001,0.01,size=N))
    now=time.mktime(time.localtime())
    day_secs = 24*3600
    times=np.linspace(now,now+N*day_secs,N)
    InvestmentPlot(investment,times,plotpath= TMP_PLOT + 'investment_test.png',title='Inv')
    RankHistPos(investment,plotpath= TMP_PLOT + 'investment_rank_hist.png')
    PnlPlot(pnl,market,times,plotpath= TMP_PLOT + 'pnl_test.png',title='P & M')
    plot_monthly_returns(times,pnl,plotpath= TMP_PLOT + 'mret_test.png')
    plot_monthly_return_distrib(times,pnl,plotpath= TMP_PLOT + 'mdist_test.png')
    plot_daily_return_underwater(times, pnl, plotpath=TMP_PLOT + 'mdd_test.png')

def XYPlot(xy, plotpath,logx=False, logy=False, lims=None, title='XY',scatter=False, frac_last = 0.1, x_label = 'x', y_label='y'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    x,y = (xy[0], xy[1])
    ord = np.argsort(x)
    x,y = (x[ord],y[ord])
    if scatter:
        plt.scatter(np.asarray(x), np.asarray(y))
    elif logx and logy:
        N = int(len(x)*frac_last)
        l = np.log(x[-N:])
        t = np.log(y[-N:])
        p = np.polyfit(l,t,  1)
        lab = '$\mu=%.2f$' % (p[0])
        pylab.loglog(x[-N:],y[-N:], color="red", linewidth=1., linestyle="-",label=lab)
    else:
        if logy:
            pylab.semilogy(x,y, color="red", linewidth=1., linestyle="-")
        else:
            pylab.plot(x,y, color="red", linewidth=1., linestyle="-")
    if not lims is None:
        plt.axis(lims)
    plt.title(r'$%s$'%title)
    plt.xlabel(r'$%s$'%x_label)
    plt.ylabel(r'$%s$'%y_label)
    pylab.legend(loc='upper right')
    pylab.savefig(plotpath, dpi=500)
    pylab.close()
    return plotpath

def MultiXYPlot(data, plotpath, logx=True, logy=True, title='XY',scatter=False, xlabel ='x', ylabel='y',frac_last = 0.1,num_subsamples=1000):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    for (name, x,y) in data:
        try:
            ord = np.argsort(x)
            x,y = (x[ord],y[ord])
            if scatter:
                plt.scatter(np.asarray(x), np.asarray(y),label=name)
            elif logx and logy:
                N = int(len(x)*frac_last)
                l = np.log(x[-N:])
                t = np.log(y[-N:])
                p = np.polyfit(l,t,  1)
                lab = '$%s:W=%d, %.3f*ds**(%.3f)$' % (name,N, np.exp(p[1]),p[0])
                ll,tt,ee  = SubSampleWithErr(l,t,num_subsamples)
                plt.errorbar(ll, tt, yerr=ee, fmt='o', elinewidth=0.2)
                l01 = [ll[0], ll[-1]]
                p01 = [p[1] + p[0] * ll[0], p[1] + p[0] * ll[-1]]
                pylab.plot(l01, p01, linestyle="--", label=lab)
            else:
                if logy:
                    pylab.semilogy(x,y,  linewidth=1., linestyle="-",label=name)
                else:
                    pylab.plot(x,y, linewidth=1., linestyle="-",label=name)
        except:
            pass
    plt.xlabel(r'$%s$'%xlabel)
    plt.ylabel(r'$%s$'%ylabel)
    plt.title(r'$%s$'%title)
    pylab.legend(loc='best')
    pylab.savefig(plotpath, dpi=500)
    pylab.close()
    return plotpath


def MultiPlot(data, plotpath, logy=False, title='XY',scatter=False, xlabel ='x', ylabel='y'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    for (name, x) in data:
        pylab.plot(x,  linewidth=1., linestyle="-",label= name)
    plt.xlabel(r'$%s$'%xlabel)
    plt.ylabel(r'$%s$'%ylabel)
    plt.title(r'$%s$'%title)
    pylab.legend(loc='upper right')
    pylab.savefig(plotpath, dpi=500)
    pylab.close()
    return plotpath

def Plot(t, plotpath='/tmp/plot.png',x_label='x',y_label='y',title='Plot'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(np.asarray(t).flatten())
    plt.xlabel(r'$%s$'%x_label)
    plt.ylabel(r'$%s$'%y_label)
    plt.title(r'$%s$'%title)
    plt.savefig(plotpath, dpi=150)
    plt.clf()

def test_plot():
    x = np.arange(100)
    np.save(TMP_PLOT + 'x.npy', x)
    np.save(TMP_PLOT + 'y.npy', x)
    PlotFiles(TMP_PLOT + 'x.npy', TMP_PLOT + 'y.npy')

def PlotTimed(times, t,  name,  plotpath):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab


    pylab.plot(np.asarray(times),np.asarray(t), color="blue", linewidth=0.5, linestyle="-", label=name)
    pylab.legend(loc='upper left')
    pylab.savefig(plotpath, dpi=300)
    pylab.close()

def Plot2(t1, t2, name1, name2,x_label,y_label, plotpath,title='Plot2',  which_corner='best'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(np.asarray(t1), color="blue", linewidth=.2, linestyle="-", label=name1)
    plt.plot(np.asarray(t2), color="red", linewidth=.2, linestyle="-", label=name2)
    plt.legend(loc=which_corner)
    plt.xlabel(r'$%s$'%x_label)
    plt.ylabel(r'$%s$'%y_label)
    plt.title(r'$%s$'%title)
    plt.savefig(plotpath, dpi=300)
    plt.clf()
    return plotpath

def PlotXYErrBars(x, y, yerr, plotpath,title='ErrBars'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    import matplotlib.pyplot as plt
    plt.errorbar(x, y, yerr=yerr, fmt='o',elinewidth=0.2)
    plt.title(r'$%s$'%title)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()
    return plotpath

def PlotErrBars(y, yerr, plotpath,title='ErrBars'):
    return PlotXYErrBars(np.arange(len(y),dtype=y.dtype),y,yerr,plotpath,title)

def test_ErrorBars():
    y = np.arange(10,dtype=float)
    err = np.ones(10,dtype=float)*0.5
    PlotErrBars(y,err,TMP_PLOT + 'test_erbar.png')

def ColorPlotMatrix(z,x_label,y_label,z_label,  plotpath):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    assert isinstance(z,np.ndarray)
    nrow,ncol = z.shape
    y, x = np.mgrid[slice(0, ncol,1),
                slice(0, nrow,1)]
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # plt.subplot(2, 1, 1)
    im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    plt.colorbar()
    # set the limits of the plot to the limits of the data
    plt.axis([x.min(), x.max()+1, y.min(), y.max()+1])
    plt.xlabel(r'$%s$'%x_label)
    plt.ylabel(r'$%s$'%y_label)
    plt.title(r'$%s$'%z_label)


    # plt.subplot(2, 1, 2)
    # # contours are *point* based plots, so convert our bound into point
    # # centers
    # plt.contourf(x + 0.5,y + 0.5, z, levels=levels,
    #              cmap=cmap)
    # plt.colorbar()
    # plt.title(z_label +' with levels')
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    return plotpath

def Plot3(times, t1, t2, name1, name2, plotpath,show_grid=False,scatter=0,width = 0.5):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    sizes1 = np.zeros(len(t1),dtype=np.int32)
    sizes1.fill(50)
    sizes2 = np.zeros(len(t2), dtype=np.int32)
    sizes2.fill(50)

    if  scatter ==0 :
        pylab.scatter(np.asarray(times), np.asarray(t1), c="blue", s=sizes1,marker='.')
        pylab.plot(np.asarray(times), np.asarray(t2), color="red", linewidth=width, linestyle="-", label=name2)
    elif scatter ==1:
        pylab.plot(np.asarray(times), np.asarray(t1), color="blue", linewidth=width, linestyle="-", label=name1)
        pylab.scatter(np.asarray(times), np.asarray(t2), c="red", s=sizes2,marker='.')
    elif scatter == 2:
        pylab.scatter(np.asarray(times), np.asarray(t1), c="blue", s=sizes1, marker='.')
        pylab.scatter(np.asarray(times), np.asarray(t2), c="red", s=sizes2, marker='.')
    else:
        pylab.plot(np.asarray(times), np.asarray(t1), color="blue", linewidth=width, linestyle="-", label=name1)
        pylab.plot(np.asarray(times), np.asarray(t2), color="red", linewidth=width, linestyle="-", label=name2)
    if show_grid:
        pylab.grid(True)
    # pylab.legend(loc='upper left')
    pylab.savefig(plotpath, dpi=1200)
    pylab.close()
    system(plotapp + " "  + plotpath)

def Plot4(times, t1, t2, t3, name1, name2, name3, plotpath,which_corner='lower right'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab


    pylab.plot(np.asarray(times),np.asarray(t1), color="blue", linewidth=0.5, linestyle="-", label=name1)
    pylab.plot(np.asarray(times),np.asarray(t2), color="red", linewidth=0.5, linestyle="-", label=name2)
    pylab.plot(np.asarray(times), np.asarray(t3), color="green", linewidth=0.5, linestyle="-", label=name3)
    pylab.legend(loc=which_corner)
    pylab.savefig(plotpath, dpi=300)
    pylab.close()
    system(plotapp + " "  + plotpath)


def PlotN(times,names, data,plotpath):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab

    for l in range(len(names)):
        pylab.plot(times, data[l], label=names[l])
    pylab.legend()
    pylab.savefig(plotpath, dpi=300)
    pylab.close()

def PlotFit(x, y, fit,plotpath,  name1='x', name2='y', title='PlotFit'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    pylab.plot(np.asarray(x),np.asarray(y), color="blue", linewidth=0.1, linestyle=':', label=name1)
    pylab.plot(np.asarray(x),np.asarray(fit), color="red", linewidth=1, linestyle='-', label=name2)
    pylab.legend(loc='upper left')
    plt.title(r'$%s$'%title)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()
    system(plotapp + " "  + plotpath)


def PlotFit2(x, y,x_fit, fit,plotpath,  name1='data', name2='fit', title='PlotFit'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    pylab.plot(np.asarray(x),np.asarray(y), color="green", linewidth=0.5, linestyle=':', label=name1)
    pylab.plot(np.asarray(x_fit),np.asarray(fit), color="red", linewidth=1, linestyle='-', label=name2)
    pylab.legend(loc='upper left')
    plt.title(r'$%s$'%title)
    pylab.savefig(plotpath, dpi=250)
    pylab.close()

def FastPlotFit(x,y,e,fit,title,name1, name2,plotpath):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    plt.errorbar(x, y, yerr=e, fmt='-o')
    pylab.plot(x,fit,color="red", linewidth=1, linestyle='-', label='fit')
    plt.xlabel(r'$%s$'%name1)
    plt.ylabel(r'$%s$'%name2)
    plt.grid()
    plt.title(r'$%s$'%title)
    pylab.savefig(plotpath, dpi=250)

    pylab.close()

def Pade(x, c1,c2,c3,c4):
    x2 = x*x
    return x*(x2*c3+c2)/(x2*c1+1) + c4

def SubSampleOne(X,K):
    N = len(X)
    grid = np.arange(K,dtype=np.int32)*(N/K)
    x = np.zeros(K,dtype=float)
    for k in range(K):
        beg = grid[k]
        end = grid[k+1] if k +1 < K else N
        if end< beg+1: continue
        x[k] = X[beg:end].mean()
    return x

def SubSample(X,Y,K):
    N = len(X)
    grid = np.arange(K,dtype=np.int32)*(N/K)
    x = np.zeros(K,dtype=float)
    y = np.zeros(K,dtype=float)
    for k in range(K):
        beg = grid[k]
        end = grid[k+1] if k +1 < K else N
        if end< beg+1: continue
        x[k] = X[beg:end].mean()
        y[k] = Y[beg:end].mean()
    return x,y

def SubSampleWithErr(X,Y,K):
    N = len(X)
    if N < 10* K:
         return X,Y,np.zeros_like(Y)
    grid = np.arange(K,dtype=float)*(float(N)/K)
    x = np.zeros(K,dtype=float)
    y = np.zeros(K,dtype=float)
    s = np.zeros(K,dtype=float)
    for k in range(K):
        beg = int(grid[k])
        end = int(grid[k+1]) if k +1 < K else N
        if end< beg+2: continue
        x[k] = X[beg:end].mean()
        y[k] = Y[beg:end].mean()
        s[k] = Y[beg:end].std()/np.sqrt(end-beg)
    if (s==0).sum() > K/2:
        s = None
    return x,y, s

def ExpGridWithErr(X,Y,K,min_num=100.):
    N = len(X)
    #exp(-(K-1)delta) = min_num
    step = float(N)/K
    min_num = min(min_num,step)
    delta = np.log(min_num/step)/(K-1.)
    steps = step*np.exp(np.arange(0,K,dtype=float)*delta)
    grid = np.cumsum(steps)
    grid -= step
    grid *= (N-1)/grid[-1]
    grid = grid.astype(np.int32)
    x = np.zeros(K,dtype=float)
    y = np.zeros(K,dtype=float)
    s = np.zeros(K,dtype=float)
    for k in range(K):
        beg = grid[k]
        end = grid[k+1] if k +1 < K else N
        if end< beg+2: continue
        x[k] = X[beg:end].mean()
        y[k] = Y[beg:end].mean()
        s[k] = Y[beg:end].std()/np.sqrt(end-beg)
    if (s==0).sum() > K/2:
        s = None
    return x,y, s

def UniformGridWithErr(X,Y,K,min_step=2):
    N = len(X)
    grid = np.searchsorted(X,np.linspace(X[0],X[-1],K))
    x = np.zeros(K,dtype=float)
    y = np.zeros(K,dtype=float)
    s = np.zeros(K,dtype=float)
    for k in range(K):
        beg = grid[k]
        end = grid[k+1] if k +1 < K else N
        if end< beg+min_step: continue
        x[k] = X[beg:end].mean()
        y[k] = Y[beg:end].mean()
        s[k] = Y[beg:end].std()/np.sqrt(end-beg)
    if (s==0).sum() > K/2:
        s = None
    return x,y, s
def PlotPredictions(predictions, results, dir, name):
    err = (predictions-results).abs().sum()/(predictions.abs()+results.abs()).sum()
    ord = predictions.copy().argsort()
    x,y,s = SubSampleWithErr(predictions[ord].copy_to_ndarray(),results[ord].copy_to_ndarray(),20)
    FastPlotFit(x=x,y=y,e=s,fit=x,
                title=name+'_prediction rel |err|=%.2f'%err,
          name1='predictions',name2='results',
          plotpath=dir + name + '_Predictor.png')

def PlotSignPredictions(predictions, results, dir, name):
    ord = predictions.copy().argsort()
    predictions = predictions[ord]
    results = results[ord]
    test = results*predictions.sign()
    strength = predictions.abs()
    x,y,s = SubSampleWithErr(strength.copy_to_ndarray(),test.copy_to_ndarray(),100)
    FastPlotFit(x=x,y=y,e=s,fit=x,
                title=name+'_strength',
          name1='strength',name2='sign*sign',
          plotpath=dir + name + '_Signs.png')

def Linear(x,a,b):
    return x*b + a

# from scipy.optimize import curve_fit
# def test_PlotFit(N, K, FitFunction=Pade,mu=7,noise=0.5):
#     #scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, **kw)[source]
#
#     x = np.random.standard_t(df=mu,size=N)  + np.random.normal(size=N)*noise
#     x = x[x!=0]
#     x = np.sort(x)
#     N = len(x)
#     cdf = np.arange(1,N+1,dtype=float)/(N+1)
#     logit = np.log(cdf/(1-cdf))
#     x,y, s = SubSampleWithErr(x,logit,K)
#     # s = None
#     C, pcov = curve_fit(FitFunction,x,y,sigma=s)#,p0=(0,0,0,0))
#     f = FitFunction(x,*C)
#     nu = C[0] if FitFunction is Linear else C[2]/C[0] if FitFunction is Pade else 0
#     title = "\\mu=%.2f;\\  \\nu=%.2f"%(mu,nu)
#     FastPlotFit(x=x,y=y,e=s,fit=f,name1='x',name2='y',title=title,plotpath=TMP_PLOT+'linear_test_fit.png')
#
#
# def SimpleFitTest():
#     x = np.linspace(-1.5,1.5,num=100)
#     y = np.tan(x)
#     def Func(x, A,B):
#         return x*A + x**3 *B/(0.25*np.pi**2 - x**2)
#     C, pcov = curve_fit(Func,x,y)
#     f = Func(x,*C)
#     FastPlotFit(x=x,y=y,e=None,fit=f,name1='x',name2='y',title="simple",plotpath=TMP_PLOT+'simple_test_fit.png')
#     XYPlot([x,y],plotpath=TMP_PLOT+'xy_plot_test.png')


def ProbHist(y, bins, plotpath, name='H',var='x'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    #pylab.loglog(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.xlabel('$%s$'%var)
    plt.ylabel('Probability')
    plt.title(r'$%s$'%name)
    plt.axis([0, 1, 0, 1.1])
    plt.grid(True)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()
    return plotpath


def CDFPlot(data, plotpath, name='CDF',var='x', lims=None, logit=False):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dir = path.dirname(plotpath)
    MakeDir(dir)
    data = np.sort(data)
    cdf = np.arange(1,len(data)+1,dtype=float)/(1.+len(data))
    if logit:
        cdf = np.log(cdf/(1-cdf))
        lims = None
        pylab.plot(data, cdf, color="black", linewidth=1., linestyle="-")
    else:
        pylab.semilogy(data, cdf, color="black", linewidth=1., linestyle="-")
    plt.xlabel('$%s$'%var)
    plt.ylabel('CDF')
    plt.title(r'$%s$'%name)
    plt.grid(True)
    if not lims is None: plt.axis(lims)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()
    return plotpath

def Hist(t, plotpath, numbins=100):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    hist, bins = np.histogram(a=t, bins=numbins, weights=None)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()


def DHist(t, plotpath, numbins=100):
    return Hist(np.diff(t),plotpath,numbins)

def Hist2(t, w, plotpath, numbins=100):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    hist, bins = np.histogram(a=t, bins=numbins, weights=w)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()
    system(plotapp + " "  + plotpath)
    return plotpath

def LogitRankHist(data,plotpath, name='LogitRankHist',var='x'):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    m,s = np.mean(data), np.std(data)
    data = (data-m)/s
    data = np.sort(data)
    data = np.sign(data)*np.log(1+np.abs(data))
    cdf = np.arange(1,len(data)+1,dtype=float)/(len(data)+1)
    cdf = np.log(cdf/(1-cdf))
    tail = cdf < -1
    mu1 = np.polyfit(data[tail], cdf[tail],1)[0]
    tail = cdf > 1
    mu2 = np.polyfit(data[tail], cdf[tail],1)[0]
    plt.xlabel(r'$sgn(%s)\log(1+|%s|)$'%(var,var))
    plt.ylabel('$\log(P(%s)/(1-P(%s))'%(var,var))
    # plt.axis([-6, 6, -6, 6])
    plt.grid(True)
    pylab.plot(data,cdf,  color="black", linewidth=1., linestyle="-")
    plt.title(r"$<%s> = %.2f\pm%.2f; \mu_<=%.2f, \mu_>=%.2f$"%(name,m,s,mu1,mu2))
    pylab.savefig(plotpath, dpi=150)
    pylab.close()


def RankHistPos(data,plotpath,name='RankHist',var_name='\eta',logx=False, logy=True):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    # dir = path.dirname(plotpath)
    # MakeDir(dir)
    # filename = plotpath.replace('.png','.RankHistPos')
    # with open(filename,'wb') as output:
    #         params = dict((
    #             ('data', data),
    #             ('plotpath',plotpath),
    #             ('name',name),
    #             ('var_name',var_name),
    #             ('logx',logx),
    #             ('logy',logy)
    #             ))
    #         pickle.dump(params,output,protocol=pickle.HIGHEST_PROTOCOL)

    x = np.sort(data[data>0])
    N = len(x)
    tail = (1-np.arange(1,N+1,dtype=float)/(N+1))
    m = x.mean()
    mean,err = data.mean(), data.std()
    gen_lab = '$%s;<%s>=%.4f\pm%.4f$; '%(name,var_name,mean,err)
    if logx:
        ok = (tail < 0.1) &  (tail > 5./N)
        l = np.log(x[ok])
        t = np.log(tail[ok])
        p = np.polyfit(l,t,  1)
        lab = '$\mu=%.2f$' % (p[0])
        pylab.loglog(x[ok],tail[ok], color="red", linewidth=1., linestyle="-",label="data")
        l01 = [np.exp(l[0]),np.exp(l[-1])]
        p01 = [np.exp(p[1] + p[0]*l[0]),np.exp(p[1] + p[0] *l[-1])]
        pylab.loglog(l01, p01, color="green", linestyle="--", label="fit "+lab)
    else:
        if logy:
            pylab.semilogy(x,tail, color="red", linewidth=1., linestyle="-")
        else:
            pylab.plot(x,tail, color="red", linewidth=1., linestyle="-")


    pass
    pylab.legend(loc='upper right')
    pylab.title(gen_lab)
    pylab.savefig(plotpath, dpi=350)
    pylab.close()
    return plotpath


def MultiRankHistPos(arrays,plotpath,var_name='\eta',logx=False, logy=True, num_subsamples=1000):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    for name, data in arrays:
        try:
            x = np.sort(data[data>0])
            N = len(x)
            tail = (1-np.arange(1,N+1,dtype=float)/(N+1))
            m = x.mean()
            mean,err = data.mean(), data.std()
            gen_lab = '$%s;<%s>=%.4f\pm%.4f$; '%(name,var_name,mean,err)
            if logx:
                ok = (tail < 0.1) &  (tail > 2./N)
                l = np.log(x[ok])
                t = np.log(tail[ok])
                p = np.polyfit(l,t,  1)
                lab = '$\mu=%.2f$' % (p[0])
                ll, tt, terr = SubSampleWithErr(l, t, num_subsamples)
                plt.errorbar(ll, tt, yerr=terr, fmt='o', elinewidth=0.2,label=name)
                l01 = [ll[0],ll[-1]]
                p01 = [p[1] + p[0]*ll[0],p[1] + p[0] *ll[-1]]
                pylab.plot(l01, p01, color="green", linestyle="--", label=name+" fit "+lab)
            else:
                if logy:
                    xx, tt, terr = SubSampleWithErr(x, np.log(tail), num_subsamples)
                    plt.errorbar(xx,tt, yerr=terr,  linewidth=1., linestyle="-",label=name)
                else:
                    xx, tt, terr = SubSampleWithErr(x, tail, num_subsamples)
                    plt.errorbar(xx,tt, yerr=terr,  linewidth=1., linestyle="-",label=name)
            pass
        except:
            pass
    pass
    pylab.legend(loc='best')
    pylab.title(gen_lab)
    pylab.savefig(plotpath, dpi=500)
    pylab.close()
    return plotpath

def RankHist2(data, plotpath, name='RankHist', var_name='\\eta', logx=False, logy=True):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dir = path.dirname(plotpath)
    MakeDir(dir)
    x1= np.sort(-data[data<=0])
    x2 = np.sort(data[data>0])
    N1 = len(x1)
    N2 = len(x2)
    W1 = N1/float(N1+N2)
    W2 = N2/float(N1+N2)
    tail1 = W1*(1-np.arange(1,N1+1,dtype=float)/(N1+1))
    tail2 = W2*(1 -np.arange(1,N2+1,dtype=float)/(N2+1))
    m1 = x1.mean()
    m2 = x2.mean()
    mean,err = data.mean(), data.std()
    gen_lab = '$%s;<%s>=%.4e\pm %.4e$; '%(name,var_name,mean,err)
    if logx:
        large = tail1 < 0.1
        l = np.log(x1[large])
        t = np.log(tail1[large])
        p1 = np.polyfit(l,t,  1)
        lab1 = '$\mu=%.4f$'%(p1[0])
        large = tail2 < 0.1
        l = np.log(x2[large])
        t = np.log(tail2[large])
        p2 = np.polyfit(l,t,  1)
        lab2 = '$\mu=%.4f$'%(p2[0])
        pylab.loglog(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
        pylab.loglog(x2,tail2, color="green", linewidth=1., linestyle="-", label=lab2)
    else:
        lab1 = '$P(%s \leq 0)=%.4f;<|%s|>=%.4f$'%(var_name,W1,var_name,m1)
        lab2 = '$P(%s > 0)=%.4f;<|%s|>=%.4f$'%(var_name,W2,var_name,m2)
        if logy:
            pylab.semilogy(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
            pylab.semilogy(x2,tail2, color="green", linewidth=1., linestyle="-", label=lab2)
        else:
            pylab.plot(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
            pylab.plot(x2,tail2, color="green", linewidth=1., linestyle="-", label=lab2)

    pass
    pylab.legend(loc='upper right')
    pylab.title(gen_lab)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()


def RankHist2WithWeights(data,weights,plotpath,name='RankHist',var_name='\eta',logx=False, logy=True):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dir = path.dirname(plotpath)
    MakeDir(dir)

    filename = plotpath.replace('.png','.RankHist2')
    with open(filename,'wb') as output:
            params = dict((
                ('data', data),
                ('plotpath',plotpath),
                ('name',name),
                ('var_name',var_name),
                ('logx',logx),
                ('logy',logy)
                ))
            pickle.dump(params,output,protocol=pickle.HIGHEST_PROTOCOL)
    pos = data>=0
    neg = ~pos
    x1 = -data[neg]
    w1 = weights[neg]
    ord1 = np.argsort(x1)
    x1= x1[ord1]
    w1 = w1[ord1]
    x2 = data[pos]
    w2 = weights[pos]
    ord2 = np.argsort(x2)
    x2 = x2[ord2]
    w2 = w2[ord2]
    w1s = w1.sum()
    w2s = w2.sum()
    WW = w1s + w2s
    W1 = w1s/WW
    W2 = w2s/WW
    tail1 = (w1s-w1.cumsum())/WW
    tail2 = (w2s-w2.cumsum())/WW
    m1 = x1.dot(w1)/w1s
    m2 = x2.dot(w2)/w2s
    mean = data.dot(weights)/WW
    gen_lab = '$%s;<%s>=%.4e$; '%(name,var_name,mean)
    if logx:
        large = tail1 < 0.01
        l = np.log(x1[large])
        t = np.log(tail1[large])
        p1 = np.polyfit(l,t,  1)
        lab1 = '$\mu=%.4f$'%(p1[0])
        large = tail2 < 0.01
        l = np.log(x2[large])
        t = np.log(tail2[large])
        p2 = np.polyfit(l,t,  1)
        lab2 = '$\mu=%.4f$'%(p2[0])
        pylab.loglog(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
        pylab.loglog(x2,tail2, color="green", linewidth=1., linestyle="-", label=lab2)
    else:
        lab1 = '$P(%s \leq 0)=%.4f;<|%s|>=%.4f$'%(var_name,W1,var_name,m1)
        lab2 = '$P(%s > 0)=%.4f;<|%s|>=%.4f$'%(var_name,W2,var_name,m2)
        if logy:
            pylab.semilogy(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
            pylab.semilogy(x2,tail2, color="green", linewidth=1., linestyle="-", label=lab2)
        else:
            pylab.plot(x1,tail1, color="red", linewidth=1., linestyle="-", label=lab1)
            pylab.plot(x2,tail2, color="green", linewidth=1., linestyle="-", label=lab2)

    pass
    pylab.legend(loc='upper right')
    pylab.title(gen_lab)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()
    return plotpath
def LogitHist(data,plotpath, fit_power=False):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pylab
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    dir = path.dirname(plotpath)
    MakeDir(dir)
    data = np.sort(data)
    l = np.arange(1,len(data)+1,dtype=float)/(1.+len(data))
    l /= 1-l
    l = np.log(l)
    z = data
    lw = 1.
    if fit_power:
        p = np.polyfit(l , z, 3)
        p2 = np.polyfit(l , z, 1)
        mess = ' data  fitted by polynomial(l)'
        pylab.plot(l,z, color="blue", linewidth=lw, linestyle="-", label='z')
        pylab.plot(l, p2[1] + l*p2[0] , color="green", linewidth=lw, linestyle="-", label='%.2f + %.2f l'%(p2[1],p2[0]))
        desc = (p[3],p[2],p[1],p[0])
        pylab.plot(l, p[3] + l * p[2] + l**2 *p[1] + l**3 *p[0], color="red", linewidth=lw, linestyle="-", \
                   label='%.2f + %.2f l +%.2f l^2 + %.2f l^3'% desc)
        pylab.legend(loc='upper left')
        pylab.title(mess)
    else:
        pylab.plot(l, z)
    pylab.savefig(plotpath, dpi=150)
    pylab.close()


def VWHist(vw, plotpath, numbins=100):
    return Hist2(vw[0], vw[1], plotpath, numbins)


def XYPlotFile(dataFile):
    if dataFile is None: return None
    if not path.isfile(dataFile): return None
    base = path.basename(dataFile)
    name = path.splitext(base)[0]
    plotpath = path.join(TMP_PLOT, name + '_XYplot.png')
    return XYPlot(np.load(dataFile), plotpath)


def VWHistFile(dataFile):
    if not path.isfile(dataFile): return None
    base = path.basename(dataFile)
    name = path.splitext(base)[0]
    plotpath = path.join(TMP_PLOT, name + '_VW_Hist.png')
    return VWHist(np.load(dataFile), plotpath)


def PlotFile(dataFile):
    if not path.isfile(dataFile): return None
    base = path.basename(dataFile)
    name = path.splitext(base)[0]
    plotpath = path.join(TMP_PLOT, name + '_plot.png')
    return Plot(np.fromfile(dataFile), plotpath)


def PlotFiles(dataFile1, dataFile2):
    if not path.isfile(dataFile1): return None
    if not path.isfile(dataFile2): return None
    base1 = path.basename(dataFile1)
    name1 = path.splitext(base1)[0]
    base2 = path.basename(dataFile2)
    name2 = path.splitext(base2)[0]
    plotpath = path.join(TMP_PLOT, name1 + '+' + name2 + '_plot.png')
    return Plot2(np.fromfile(dataFile1), np.fromfile(dataFile2), name1, name2, plotpath)


def HistFile(dataFile):
    if not path.isfile(dataFile): return None
    base = path.basename(dataFile)
    name = path.splitext(base)[0]
    plotpath = path.join(TMP_PLOT, name + '_hist.png')
    return Hist(np.fromfile(dataFile), plotpath)

def DHistFile(dataFile):
    if not path.isfile(dataFile): return None
    base = path.basename(dataFile)
    name = path.splitext(base)[0]
    plotpath = path.join(TMP_PLOT, name + '_hist.png')
    return DHist(np.fromfile(dataFile), plotpath)

def HistFiles(dataFile, weightFile):
    if not path.isfile(dataFile): return None
    if not path.isfile(weightFile): return None
    base = path.basename(dataFile)
    name = path.splitext(base)[0]
    plotpath = path.join(TMP_PLOT, name + '_weighted_hist.png')
    return Hist2(np.fromfile(dataFile), np.fromfile(weightFile), plotpath)


def usage():
    print("Usage: %s -h|-p|-xy <filename(s)>" % str(sys.argv[0]))
    print(" -h - generate histogram(weighted)")
    print(" -p - generate plot(s)")
    print(" -xy - generate XYplot")
    print(" -vw - generate VWhist")


def test_SemiLog(N=1000):
    import os
    x1 = np.random.pareto(a=2,size=N)
    dir = TMP_PLOT + 'movies/pareto/'
    MakeDir(dir)
    for n in range(5):
        a = (2.+n)/2.
        x2 = np.random.pareto(a=a,size=int(N*1.1))
        x = np.append(-x1,x2)
        title = 'P_{>}(\eta | P_{>}(%s)=%.2f)'%('indicator',1-2**-n)
        RankHist2(x, dir + 'frame_%d.png'%n,name=title,logx=True)


def test_LogitHist(N=1000, fit =False):
    x = np.random.standard_t(df=2,size=N)
    LogitRankHist(x,plotpath=TMP_PLOT + 'logitplot_student2.png',name='student2')
    x = np.random.standard_t(df=5,size=N)
    LogitRankHist(x,plotpath=TMP_PLOT + 'logitplot_student5.png',name='student5')
    x = np.random.normal(size=N)
    LogitRankHist(x,plotpath=TMP_PLOT + 'logitplot_normal.png',name='normal')
    x = np.random.uniform(size=N,low=-1,high=1)
    LogitRankHist(x,plotpath=TMP_PLOT + 'logitplot_uniform.png',name='uniform')

def test_RankHist2():
    N = 100000
    data = np.random.normal(size=N)
    RankHist2(data, TMP_PLOT + 'test_mpl.png')


def test_VectoHist(N=100):
    alpha = np.linspace(-np.pi/2,np.pi/2,N)
    y = np.cos(alpha)
    bins = np.linspace(0.,1.,N)
    ProbHist(y, bins, TMP_PLOT + 'cos_hist.png', name='Cos',var='x')


def test_CDF(N=10000):
    import os
    dir = TMP_PLOT + 'cdf/uniform+power/'
    MakeDir(dir)
    x = np.random.uniform(size=N)
    r = np.random.uniform(size=N)
    y = x**r
    base = y[np.argsort(y)]
    CDFPlot(y, dir + 'base.png',name='CDF',var='x', lims=[0,1,0,1])
    K = 10
    for n in range(1,K+1):
        q = n/float(K)
        condition = r<q
        cond_set = x[condition]**r[condition]
        mm = np.searchsorted(base,cond_set)
        cdf = (mm.astype(float) +1)/(N+1)
        M = len(mm)
        title = 'q=%s'%q
        CDFPlot(cdf, dir + 'frame_%d.png'%n,name=title,var='C0',lims=[0,1,0,1])

def test_CDFSaving(N=10000):
    import os
    dir = TMP_PLOT + 'cdf/uniform+power/'
    MakeDir(dir)
    x = np.random.uniform(size=N)
    r = np.random.uniform(size=N)
    y = x**r
    CDFPlot(y, dir + 'base.png',name='CDF',var='x', lims=[0,1,0,1])

SavedFunctions = dict()
SavedFunctions["CDFPlot"] = CDFPlot
SavedFunctions["Plot2"] = Plot2
SavedFunctions["XYPlot"] = XYPlot
SavedFunctions["LogitRankHist"] = LogitRankHist
SavedFunctions["RankHist2"] = RankHist2
SavedFunctions["RankHistPos"] = RankHistPos

def ViewSaved(filename):
    ext = filename.split('.')[-1]
    F = SavedFunctions.get(ext)
    if F is None:
        print("unsupported extension %s" % ext)
        return filename
    try:
        with open(filename,'r') as input:
            params = pickle.load(input)
            params['plotpath'] = filename.replace(ext,'png')
            plotpath = F(**params)
            if not plotpath is None:
                os.remove(filename)
                return plotpath
            else:
                return filename
    except Exception as ex:
        print('Exception in ViewSaved', ex)
        return filename


def ViewSavedFiles(pattern):
    paths = []
    for filename in glob.glob(pattern):
        if os.path.isdir(filename):
            paths.append(ViewSavedFiles(os.path.join(filename,'*')))
        else:
            paths.append(ViewSaved(filename))
    return paths

def ViewSavedFilesInFolder(path):
    bb = os.path.basename(path).split('.')[0]
    return ViewSavedFiles(path.replace(bb,'*'))

