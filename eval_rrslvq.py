from __future__ import division

import os
import sys

import numpy as np
import pandas as pd

from bix.classifiers.rrslvq import RRSLVQ
from bix.classifiers.rslvq import RSLVQ
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from bix.evaluation.crossvalidation import CrossValidation
from bix.evaluation.gridsearch import GridSearch
from bix.utils.geometric_median import *


from skmultiflow.lazy.knn import KNN
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.meta import OzaBaggingAdwin
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.trees.hoeffding_adaptive_tree import HAT


def parameter_grid_search():
        grid = {"sigma": np.arange(2,21,2), "prototypes_per_class": np.arange(2,11,2)}
        gs = GridSearch(RRSLVQ(),grid)
        gs.search()
        gs.save_summary()


def test_grid():
    clfs = [RRSLVQ(),RSLVQ(),HAT(),OzaBaggingAdwin(base_estimator=KNN()),AdaptiveRandomForest(),SAMKNN()]
    cv = CrossValidation(clfs=clfs,max_samples=500,test_size=1)
    cv.streams =cv.init_reoccuring_streams()
    cv.test()
    cv.save_summary()
 
if __name__ == "__main__":
        test_grid()
