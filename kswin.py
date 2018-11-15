from abc import ABCMeta, abstractmethod
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
from skmultiflow.core.base_object import BaseObject       
from scipy import stats
from scipy.stats import f_oneway
class KSWIN(BaseDriftDetector):
    def __init__(self,alpha=0.05,data=None):

        self.w_size = 100
        self.stat_size = 30 
        self.alpha = alpha
        self.change_detected = False;
        self.p_value = 0
        self.n = 0
        if type(data) != list or type(data) == None:
            self.window = []
        else:
            self.window = data
        pass
    
    def add_element(self, value):
        self.n +=1
        currentLength = len(self.window)
        if currentLength >= self.w_size:
            self.window.pop(0)
            rnd_window = np.random.choice(self.window[:-self.stat_size],self.stat_size)
            #rnd_window = self.window[:-self.stat_size]
            (st, self.p_value) = stats.ks_2samp(rnd_window, self.window[-self.stat_size:])
           
            if self.p_value <= self.alpha and 0.1 < st:
                self.change_detected = True
                self.window = self.window[-self.stat_size:]
            else: 
                self.change_detected = False
        else: 
            self.change_detected = False
        self.window.insert(currentLength,value)      
        pass
    
    def detected_change(self):
        return self.change_detected
    
    def reset(self):
        self.alpha = 0
        self.window = []
        self.change_detected = False;
        pass
    
    def get_info(self):
         return "KSwin Change: Current P-Value "+str(self.p_value)


if __name__ == "__main__":
    from random import random as rnd
    import sys
    from skmultiflow.data.sea_generator import SEAGenerator
    import numpy as np

 
    
    kswin = KSWIN(alpha=0.001)
    stream = SEAGenerator(classification_function = 2, random_state = 112, balance_classes = False,noise_percentage = 0.28)
    stream.prepare_for_use()

    stream.restart()
    detections,mean = [],[]
    
    print("\n--------------------\n")
    for i in range(10000):
        data = stream.next_sample(10)
        batch = data[0][0][0]
        mean.append(batch)
        kswin.add_element(batch)
        if kswin.detected_change():
            print("\rIteration {}".format(i))
            print("\r KSWINReject Null Hyptheses")
            print(np.mean(mean))
            mean = []
            detections.append(i)

    print(len(detections))