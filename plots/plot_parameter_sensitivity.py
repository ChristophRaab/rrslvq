import numpy as np
import pandas as pd
import math as m
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

x = np.arange(1,1000,1)

linestyles = ['-', '--', '-.', ':']

def plotalpha():
    alpha = np.arange(10,10000,1)
    alpha = 1 / alpha
    fig, ax = plt.subplots()
    wr = np.arange(100,500,100)

    for r,style in zip(wr,linestyles):
        y = [ks(a,r) for a in alpha]

        ax.plot(alpha,y,label="$r$ = "+str(r),linestyle=style)
    SIZE = 18
    SMALL = 14

    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    ax.set_xscale("log")
    ax.legend()
    ax.set(xlabel=r'$\alpha$', ylabel='Required Distance')
    plt.show()
    fig.savefig("ps_ks_alpha.eps",dpi=1000, format='eps',quality=95)
    
def plotr():
    x = np.arange(10,1000,1) 
    alphas = [0.1,0.01,0.001,0.0001]
    fig, ax = plt.subplots()

    for a,style in zip(alphas,linestyles):
        y = [ks(a,r) for r in x]
        ax.plot(x,y,label=r"$\alpha$ = "+str(a),linestyle=style)
    SIZE = 18
    SMALL = 14
    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)

    # ax.set_xscale("log")
    ax.set(xlabel='Sample Size $r$', ylabel='Required Distance') 
    ax.legend()
    plt.show()
    fig.savefig("ps_ks_r.eps",dpi=1000, format='eps',quality=95)


def ks(alpha,r):
    return m.sqrt(-1*(m.log(alpha))/(r))

plotr()
plotalpha()
plotr()



