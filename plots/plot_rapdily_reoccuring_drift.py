
from __future__ import division

import math
import copy
import sys
from random import random as rnd

from skmultiflow.data.mixed_generator import MIXEDGenerator
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.classifiers.rrslvq import RRSLVQ
from bix.classifiers.rslvq import RSLVQ
from skmultiflow.lazy.sam_knn import SAMKNN

from skmultiflow.core.pipeline import Pipeline
s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)


stream = ReoccuringDriftStream(stream=s1, drift_stream=s2,random_state=None,alpha=90.0, position=2000,width=1)

n_prototypes_per_class = 4
sigma = 10
model_names = ["rslvq","oza","ks_rslvq"]

stream.prepare_for_use()
rrslvq = RRSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.05,sigma=sigma,replace=True)
irslvq = RRSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="KS",confidence=0.001,sigma=sigma,replace=False)
arslvq = RRSLVQ(prototypes_per_class=n_prototypes_per_class,drift_detector="ADWIN",confidence=0.01,sigma=sigma,replace=True)
rslvq = RSLVQ(prototypes_per_class=n_prototypes_per_class,sigma=sigma)     
adf = SAMKNN()
pipe = Pipeline([('Classifier', rslvq)])  
classifier = [pipe,adf,rrslvq,arslvq,irslvq]
# Switch to Geometric median for stability plot
# Replace ":" with " " in evaluation_visualzies.py to remove mean from plot
evaluator = EvaluatePrequential(show_plot=True,batch_size=10,max_samples=10000,metrics=['accuracy', 'kappa','kappa_m'],    
                                output_file=None)

evaluator.evaluate(stream=stream, model=classifier,model_names=["rslvq","SamKNN","Rrslvq","Arslvq","Irslvq"])
