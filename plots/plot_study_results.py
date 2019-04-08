
from __future__ import division

import math
import copy
import sys
from random import random as rnd
sys.path.append('..\\multiflow-rslvq')
sys.path.append('..\\RSLVQ')
sys.path.append('..\\stream_utilities')

from GridSearch import GridSearch
from ReoccuringDriftStream import ReoccuringDriftStream
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
from ReoccuringDriftStream import ReoccuringDriftStream 
from kswin import KSWIN
from rrslvq import RRSLVQ as rRSLVQ
from rslvq import RSLVQ as bRSLVQ
from skmultiflow.core.base import StreamModel
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
# Incremental Concept Drift Generators
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.mixed_generator import MIXEDGenerator
# No Concept Drift Generators
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.sine_generator import SineGenerator
#Abrupt Concept Drift Generators
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.lazy.knn import KNN
from skmultiflow.meta import OzaBaggingAdwin
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from stream_rslvq import RSLVQ as sRSLVQ
import itertools
from glvq import GLVQ
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest

s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)
stream = ReoccuringDriftStream(stream=s1, drift_stream=s2,random_state=None,alpha=90.0, position=2000,width=1,pause = 1000)

n_prototypes_per_class = 4
sigma = 12
model_names = ["rslvq","oza","ks_rslvq"]
#rRSLVQ(),sRSLVQ(),HAT(),AdaptiveRandomForest(),NaiveBayes()
stream.prepare_for_use()
rrslvq = rRSLVQ(prototypes_per_class=n_prototypes_per_class,drift_handling="KS",sigma=sigma,replace=True)
rslvq = sRSLVQ(prototypes_per_class=n_prototypes_per_class,sigma=sigma)     
oza = OzaBaggingAdwin(base_estimator=KNN())
adf = AdaptiveRandomForest()

pipe = Pipeline([('Classifier', rslvq)])  
classifier = [pipe,adf,rrslvq]

evaluator = EvaluatePrequential(show_plot=True,batch_size=10,max_samples=10000,metrics=['accuracy','kappa'],    
                                output_file=None)

evaluator.evaluate(stream=stream, model=classifier,model_names=["rslvq","adf","rrslvq"])
