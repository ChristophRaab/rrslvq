import os,sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


files = ["Mixed Generator_Mixed Generator_1000_100_memory_other.csv","Mixed Generator_memory_other.csv"]
for file in files:
    names = ["hat","rslvq", "arslvq", "adf", "oza","samknn"]

    df =pd.read_csv("data/"+file,skiprows=9)


    clfs = [df["total_running_time_["+name+"]"].values for name in names]
    names = names
    fig, ax = plt.subplots()

    for clf,name in zip(clfs,names):

        print(name+ " max " +str(np.max(clf)) + " min "+str(np.min(clf)))

        ax.plot(df["id"], clf,label=name)
    SIZE = 16
    SMALL = 14
    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)

    ax.legend()
    ax.set(xlabel="Timesteps", ylabel='Memory Usage in KB')
    plt.show()
    fig.savefig("memory_usage_"+file+".eps", dpi=1000, format='eps',bbox_inches='tight')