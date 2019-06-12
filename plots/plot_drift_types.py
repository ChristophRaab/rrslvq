import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
import pandas as pd
from matplotlib.axes import Axes, Subplot
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def reoccuring_drift(length=50000,width=10,rate=0.1,plot=True,filename="reoccuring_drift.eps"):
    length = length / 2
    probability_drift = np.array([])
    inv_probability_drift = np.array([])
    time = np.array([])

    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    part_length = rate*length
    for part in range(int(length/part_length)):
        t = np.arange(time.size, time.size+part_length, 1)
        x = np.array([1.0 / (1.0 + np.exp(-4.0 * float(i - int(time.size+part_length-part_length/2)) / float(width))) for i in t])
        y = np.array([1 - p for p in x])
        probability_drift = np.append(probability_drift,x)
        probability_drift = np.append(probability_drift,y)
        time = np.append(time,t)

    probability_drift = (probability_drift-.5)*2
    
    t = np.arange(1, probability_drift.size+1, 1)
    

    signal = probability_drift
    
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan

    ax.plot(pos_signal,label="Concept 1")
    ax.plot(neg_signal,label="Concept 2")

    plot_attributes(plt,ax)

    fig.savefig(filename,dpi=1000, format='eps',quality=100,bbox_inches='tight')

    plt.show() if plot else ""


def incremental_drift(length=50000,width=10000,plot=True,filename="incremental_drift.eps"):
    probability_drift = np.array([])
    inv_probability_drift = np.array([])
    time = np.array([])

    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    t = np.arange(time.size, length, 1)
    x = np.array([1.0 / (1.0 + np.exp(-4.0 * float(i - int(length/2)) / float(width))) for i in t])
    y = np.array([1 - p for p in x])
    probability_drift = np.append(probability_drift,x)
    # probability_drift = np.append(probability_drift,y)
    time = np.append(time,t)

    probability_drift = (probability_drift-.5)*2

    t = np.arange(1, probability_drift.size+1, 1)

    signal = probability_drift
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan


    ax.plot(pos_signal,label="Concept 1")
    ax.plot(neg_signal,label="Concept 2")
    plot_attributes(plt,ax)

    fig.savefig(filename,dpi=1000, format='eps',quality=100,bbox_inches='tight')

    plt.show() if plot else ""


def gradual_drift(length=50000,width=10,rate=0.4,plot=True,filename="gradual_drift.eps"):
    length = length / 2
    probability_drift = np.array([])
    inv_probability_drift = np.array([])
    time = np.array([])

    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    part_length = rate*length
    for part in range(int(length/part_length)):

        t = np.arange(time.size, time.size+part_length, 1)
        x = np.array([1.0 / (1.0 + np.exp(-4.0 * float(i - int(time.size+part_length-part_length/2)) / float(width))) for i in t])
        y = np.array([1 - p for p in x])

        if 0 == part:
            probability_drift = np.append(probability_drift,np.zeros(10000))
        if int(length/part_length)-1 == part:
            probability_drift = np.append(probability_drift,x)   
            probability_drift = np.append(probability_drift,np.ones(10000))
        else:
            probability_drift = np.append(probability_drift,x)
            probability_drift = np.append(probability_drift,y)

        time = np.append(time,t)

    probability_drift = (probability_drift-.5)*2
    t = np.arange(1, probability_drift.size+1, 1)

    signal = probability_drift
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan
    ax.plot(pos_signal,label="Concept 1")
    ax.plot(neg_signal,label="Concept 2")
    plot_attributes(plt,ax)

    plt.show() if plot else ""
    fig.savefig(filename,edpi=1000, format='eps',quality=100)

def plot_attributes(plt,ax):
    #plotting
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Data Mean')
    plt.style.use('seaborn-paper')


    ax.legend()
    plt.yticks([-1,1.0],["Concept 1","Concept 2"],rotation='vertical')
    ticks = ax.yaxis.get_majorticklabels()
    ticks[0].set_verticalalignment("center")
    ticks[1].set_verticalalignment("center")


    # ax1 = ax.twinx()
    # plt.yticks([-1,0,1],["","",""],rotation='vertical')




reoccuring_drift(width=600,filename="frequent_reoccuing_drift.eps") # Frequent Reoccurring
reoccuring_drift(width=1000,rate=0.4) # Reoccurring
incremental_drift(width=15000) # Incremental
incremental_drift(width=2500,filename="abrupt_drift.eps") # Abrupt
gradual_drift(length=50000,width=1000,rate=0.4) #Gradual
