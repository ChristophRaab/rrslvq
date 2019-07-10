import os,sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


files = ["Mixed Generator_Mixed Generator_1000_100_memory_other.csv","Mixed Generator_memory_other.csv"]
for file in files:
    names = ["hat","rslvq", "arslvq", "adf", "oza"]

    df =pd.read_csv("data/"+file,skiprows=9)


    clfs = [df["model_size_["+name+"]"].values for name in names]
    names = names
    fig, ax = plt.subplots()

    for clf,name in zip(clfs,names):

        print(name+ " max " +str(np.max(clf)) + " min "+str(np.min(clf)))

        ax.plot(df["id"], clf,label=name)


    ax.legend()
    ax.set(xlabel="Timesteps", ylabel='Memory Usage in KB')
    plt.show()
    fig.savefig("test.eps", dpi=1000, format='eps')
# files = ["Mixed Generator_Mixed Generator_1000_100_memory_samknn.csv", "Mixed Generator_memory_samknn.csv"]
for file in files:
    names = ["samknn"]

    df =pd.read_csv("data/"+file,skiprows=9)


    clfs = [df["model_size_["+name+"]"].values for name in names]
    names = names
    fig, ax = plt.subplots()

    for clf,name in zip(clfs,names):

        print(name+ " max " +str(np.max(clf)) + " min "+str(np.min(clf)))

        ax.plot(df["id"], clf,label=name)


    ax.legend()
    ax.set(xlabel="Timesteps", ylabel='Memory Usage in KB')
    plt.show()
    fig.savefig("test.eps", dpi=1000, format='eps')