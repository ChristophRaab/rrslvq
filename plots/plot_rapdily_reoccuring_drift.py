
from __future__ import division

import math
import copy
import sys
from random import random as rnd

from skmultiflow.data.mixed_generator import MIXEDGenerator
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.classifiers.arslvq import ARSLVQ
from bix.classifiers.rslvq import RSLVQ
from bix.classifiers.rrslvq import RRSLVQ
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.lazy.knn import KNN
from skmultiflow.core.pipeline import Pipeline
s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)


stream = ReoccuringDriftStream(stream=s1, drift_stream=s2,random_state=None,alpha=90.0, position=2000,width=1)

n_prototypes_per_class = 4
sigma = 10
ReoccuringDriftStream()
stream.prepare_for_use()
ARSLVQ()
arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.01,sigma=sigma,replace=True,window_size=1000)
rrslvq = RRSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.05,sigma=sigma,replace=True)
# irslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.001,sigma=sigma,replace=False)
# arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="ADWIN",confidence=0.01,sigma=sigma,replace=True)
rslvq = RSLVQ(prototypes_per_class=n_prototypes_per_class,sigma=sigma)     
adf = SAMKNN()
pipe = Pipeline([('Classifier', rslvq)])  
classifier = [pipe,arslvq,rrslvq] #,adf,ARSLVQ,arslvq,irslvq]
# Switch to Geometric median for stability plot
# Replace ":" with " " in evaluation_visualzies.py to remove mean from plot
evaluator = EvaluatePrequential(show_plot=True,batch_size=10,max_samples=10000,metrics=['accuracy'],    
                                output_file=None)
                                
evaluator.evaluate(stream=stream, model=classifier,model_names=["rslvq","arslvq","rrslvq"]) #,"SamKNN","ARSLVQ","Arslvq","Irslvq"])
arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.05,sigma=sigma,replace=True)
# irslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.001,sigma=sigma,replace=False)
# arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="ADWIN",confidence=0.01,sigma=sigma,replace=True)
rslvq = RSLVQ(prototypes_per_class=n_prototypes_per_class,sigma=sigma)     
adf = SAMKNN()
pipe = Pipeline([('Classifier', ARSLVQ)])  
classifier = [pipe] #,adf,ARSLVQ,arslvq,irslvq]
# Switch to Geometric median for stability plot
# Replace ":" with " " in evaluation_visualzies.py to remove mean from plot
evaluator = EvaluatePrequential(show_plot=True,batch_size=10,max_samples=10000,metrics=['accuracy'],    
                                output_file=None)

evaluator.evaluate(stream=stream, model=classifier,model_names=["ARSLVQ"]) #,"SamKNN","ARSLVQ","Arslvq","Irslvq"])

oza = OzaBaggingAdwin(base_estimator=KNN())
arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.05,sigma=sigma,replace=True)
# irslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.001,sigma=sigma,replace=False)
# arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="ADWIN",confidence=0.01,sigma=sigma,replace=True)
rslvq = RSLVQ(prototypes_per_class=n_prototypes_per_class,sigma=sigma)     
adf = SAMKNN()
pipe = Pipeline([('Classifier', oza)])  
classifier = [pipe] #,adf,ARSLVQ,arslvq,irslvq]
# Switch to Geometric median for stability plot
# Replace ":" with " " in evaluation_visualzies.py to remove mean from plot
evaluator = EvaluatePrequential(show_plot=True,batch_size=10,max_samples=10000,metrics=['accuracy'],    
                                output_file=None)

evaluator.evaluate(stream=stream, model=classifier,model_names=["oza"]) #,"SamKNN","ARSLVQ","Arslvq","Irslvq"])
