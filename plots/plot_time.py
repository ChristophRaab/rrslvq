import os,sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


files = ["Abrupt_time.csv","RA_time.csv"]
names = ["Abrupt_time.eps","RA_time.eps"]
SIZE = 16
SMALL = 14
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL)  # legend fontsize
plt.rc('figure', titlesize=SIZE)
for file_name,file in zip(names,files):
    names = ["hat","rslvq", "arslvq", "adf", "oza","samknn"]

    df =pd.read_csv("data/"+file,skiprows=11)


    clfs = [df["total_running_time_["+name+"]"].values for name in names]
    names = names
    fig, ax = plt.subplots()

    for clf,name in zip(clfs,names):

        print(name+ " max " +str(np.max(clf)) + " min "+str(np.min(clf)))

        ax.plot(df["id"], clf,label=name)



    ax.legend()
    ax.set(xlabel="Processed Samples", ylabel='Time in Seconds')
    plt.show()
    fig.savefig(file_name, dpi=1000, format='eps',bbox_inches='tight')