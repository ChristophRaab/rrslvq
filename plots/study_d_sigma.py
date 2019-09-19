from __future__ import division


from bix.data.reoccuringdriftstream import ReoccuringDriftStream
import matplotlib.pyplot as plt

from skmultiflow.data.mixed_generator import MIXEDGenerator

#Abrupt Concept Drift Generators
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:35:11 2019

@author: moritz
"""



"""
Created on Wed May 29 15:35:11 2019

@author: moritz
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist
from bix.detectors.kswin import KSWIN
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn import metrics

class ARSLVQ(ClassifierMixin, BaseEstimator):
    """Reactive Robust Soft Learning Vector Quantization
    Parameters
    ----------
    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.
    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.
    sigma : float, optional (default=0.5)
        Variance for the distribution.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    drift_detector : string, Type of concept drift DETECTION.
        None means no concept drift detection
        If KS, use of Kolmogorov Smirnov test
        IF DIST, monitoring class distances to detect outlier.
    gamma : float, Decay Rate for Adadelta (default=0.9)

    Attributes
    ----------
    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes
    classes_ : array-like, shape = [n_classes]
        Array containing labels.
    initial_fit : boolean, indicator for initial fitting. Set to false after
        first call of fit/partial fit.
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, random_state=112, drift_detector="KS", confidence=0.05,
                 gamma: float = 0.9, replace: bool = True, window_size: int = 200):
        self.sigma = sigma

        self.random_state = random_state
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []

        #### Reactive extensions  ####
        self.confidence = confidence
        self.counter = 0
        self.cd_detects = []
        self.drift_detector = drift_detector
        self.drift_detected = False
        self.replace = replace
        self.init_drift_detection = True
        self.window_size = window_size

        #### Adadelta ####
        self.decay_rate = gamma
        self.epsilon = 1e-8

        if self.prototypes_per_class < 1:
            raise ValueError("Number of prototypes per class must be greater or equal to 1")

        if self.drift_detector != "KS" and self.drift_detector != "DIST" and self.drift_detector != "ADWIN":
            raise ValueError("Drift detector must be either KS, ADWIN or DIST!")

        if self.confidence <= 0 or self.confidence >= 1:
            raise ValueError("Confidence of test must be between 0 and 1!")

        if self.sigma < 0:
            raise ValueError("Sigma must be greater than zero")

    def _optimize(self, x, y, random_state):
        """Implementation of Adadelta"""
        n_data, n_dim = x.shape
        nb_prototypes = self.c_w_.size
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = x[i]
            c_xi = y[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])
                #d = metrics.pairwise.cosine_similarity(xi.reshape(-1, 1), prototypes[j].reshape(-1, 1))[0,:]
                if self.c_w_[j] == c_xi:
                    gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                self._p(j, xi, prototypes=self.w_)) * d
                else:
                    gradient = - self._p(j, xi, prototypes=self.w_) * d

                # Accumulate gradient
                self.squared_mean_gradient[j] = self.decay_rate * self.squared_mean_gradient[j] + \
                                                (1 - self.decay_rate) * gradient ** 2

                # Compute update/step
                step = ((self.squared_mean_step[j] + self.epsilon) / \
                        (self.squared_mean_gradient[j] + self.epsilon)) ** 0.5 * gradient

                # Accumulate updates
                self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + \
                                            (1 - self.decay_rate) * step ** 2

                # Attract/Distract prototype to/from data point
                self.w_[j] += step
            #            """Implementation of Stochastical Gradient Descent"""

    #            n_data, n_dim = X.shape
    #            nb_prototypes = self.c_w_.size
    #            prototypes = self.w_.reshape(nb_prototypes, n_dim)
    #
    #            for i in range(n_data):
    #                xi = X[i]
    #                c_xi = y[i]
    #                for j in range(prototypes.shape[0]):
    #                    d = (xi - prototypes[j])
    #                    c = 1/ self.sigma
    #                    if self.c_w_[j] == c_xi:
    #                        # Attract prototype to data point
    #                        self.w_[j] += c * (self._p(j, xi, prototypes=self.w_, y=c_xi) -
    #                                     self._p(j, xi, prototypes=self.w_)) * d
    #                    else:
    #                        # Distance prototype from data point
    #                        self.w_[j] -= c * self._p(j, xi, prototypes=self.w_) * d

    def _costf(self, x, w, **kwargs):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(d)
       # d = metrics.pairwise.cosine_similarity(x.reshape(1,-1),w.reshape(1,-1))
        self.sigma = x.shape[0]
        return -d / (2 * self.sigma)

    def _p(self, j, e, y=None, prototypes=None, **kwargs):
        if prototypes is None:
            prototypes = self.w_
        if y is None:
            fs = [self._costf(e, w, **kwargs) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i], **kwargs) for i in
                  range(prototypes.shape[0]) if
                  self.c_w_[i] == y]

        fs_max = max(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        o = np.math.exp(
            self._costf(e, prototypes[j], **kwargs) - fs_max) / s
        return o

    def get_prototypes(self):
        """Returns the prototypes"""
        return self.w_

    def predict(self, x):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return np.array([self.c_w_[np.array([self._costf(xi, p) for p in self.w_]).argmax()] for xi in x])

    def posterior(self, y, x):
        """
        calculate the posterior for x:
         p(y|x)
        Parameters
        ----------

        y: class
            label
        x: array-like, shape = [n_features]
            sample
        Returns
        -------
        posterior
        :return: posterior
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.column_or_1d(x)
        if y not in self.classes_:
            raise ValueError('y must be one of the labels\n'
                             'y=%s\n'
                             'labels=%s' % (y, self.classes_))
        s1 = sum([self._costf(x, self.w_[i]) for i in
                  range(self.w_.shape[0]) if
                  self.c_w_[i] == y])
        s2 = sum([self._costf(x, w) for w in self.w_])
        return s1 / s2

    def get_info(self):
        return 'RSLVQ'

    def predict_proba(self, X):
        """ predict_proba

        Predicts the probability of each sample belonging to each one of the
        known target_values.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is
            associated with the X entry of the same index. And where the list in
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.

        """
        return 'Not implemented'

    def reset(self):
        self.__init__()

    def _validate_train_parms(self, train_set, train_lab, classes=None):
        random_state = validation.check_random_state(self.random_state)
        train_set, train_lab = validation.check_X_y(train_set, train_lab.ravel())

        if (self.initial_fit):
            if (classes):
                self.classes_ = np.asarray(classes)
                self.protos_initialized = np.zeros(self.classes_.size)
            else:
                self.classes_ = unique_labels(train_lab)
                self.protos_initialized = np.zeros(self.classes_.size)

        nb_classes = len(self.classes_)
        nb_samples, nb_features = train_set.shape  # nb_samples unused

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int) or isinstance(self.prototypes_per_class, np.int64):
            if self.prototypes_per_class < 0 or not isinstance(
                    self.prototypes_per_class, int) and not isinstance(
                self.prototypes_per_class, np.int64):
                # isinstance(self.prototypes_per_class, np.int64) fixes the singleton array array (1) is ... bug of gridsearch parallel
                raise ValueError("prototypes_per_class must be a positive int")
            # nb_ppc = number of protos per class
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        else:
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    "values in prototypes_per_class must be positive")
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    "length of prototypes per class"
                    " does not fit the number of classes"
                    "classes=%d"
                    "length=%d" % (nb_classes, nb_ppc.size))

        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClassIdx in range(len(self.classes_)):
                actClass = self.classes_[actClassIdx]
                nb_prot = nb_ppc[actClassIdx]  # nb_ppc: prototypes per class
                if (self.protos_initialized[actClassIdx] == 0 and actClass in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == actClass, :], 0)
                    self.w_[pos:pos + nb_prot] = mean + (
                            random_state.rand(nb_prot, nb_features) * 2 - 1)
                    if math.isnan(self.w_[pos, 0]):
                        print('Prototype is NaN: ', actClass)
                        self.protos_initialized[actClassIdx] = 0
                    else:
                        self.protos_initialized[actClassIdx] = 1

                    self.c_w_[pos:pos + nb_prot] = actClass
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        if self.initial_fit:
            # Next two lines are Init for Adadelta/RMSprop
            self.squared_mean_gradient = np.zeros_like(self.w_)
            self.squared_mean_step = np.zeros_like(self.w_)
            self.initial_fit = False

        return train_set, train_lab, random_state

    def fit(self, X, y, classes=None):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)
        Returns
        --------
        self
        """
        X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        if len(np.unique(y)) == 1:
            raise ValueError("fitting " + type(
                self).__name__ + " with only one class is not possible")
        # X = preprocessing.scale(X)
        self._optimize(X, y, random_state)
        return self

    def partial_fit(self, X, y, classes=None):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)
        Returns
        --------
        self
        """

        if set(unique_labels(y)).issubset(set(self.classes_)) or self.initial_fit == True:
            X, y, random_state = self._validate_train_parms(
                X, y, classes=classes)
        else:
            raise ValueError(
                'Class {} was not learned - please declare all classes in first call of fit/partial_fit'.format(y))

        self.counter = self.counter + 1
        if self.drift_detector is not None and self.concept_drift_detection(X, y):
            self.cd_handling(X, y)
            self.cd_detects.append(self.counter)
        # X = preprocessing.scale(X)
        self._optimize(X, y, self.random_state)
        return self

    def save_data(self, X, y, random_state):
        pd.DataFrame(self.w_).to_csv("Prototypes.csv")
        pd.DataFrame(self.c_w_).to_csv("Prototype_Labels.csv")
        pd.DataFrame(X).to_csv("Data.csv")
        pd.DataFrame(y).to_csv("Labels.csv")
        self._optimize(X, y, random_state)
        pd.DataFrame(self.w_).to_csv("Prototypes1.csv")
        pd.DataFrame(self.c_w_).to_csv("Prototype_Labels1.csv")

    def calcDistances(self, pts, x):
        dists = []
        for p in pts:
            for elem in x:
                dists.append(np.linalg.norm(p - elem))
        return np.max(dists)

    def concept_drift_detection(self, X, Y):
        if self.init_drift_detection:
            if self.drift_detector == "KS":
                self.cdd = [KSWIN(alpha=self.confidence, w_size=self.window_size) for elem in X.T]
            if self.drift_detector == "ADWIN":
                self.cdd = [ADWIN(delta=self.confidence) for elem in X.T]
            if self.drift_detector == "DIST":
                self.cdd = [KSWIN(self.confidence, w_size=self.window_size) for c in self.classes_]
        self.init_drift_detection = False
        self.drift_detected = False

        if self.drift_detector == "DIST":
            try:
                class_prototypes = [self.w_[self.c_w_ == elem] for elem in self.classes_]
                new_distances = dict(
                    [(c, self.calcDistances(pts, X[Y == c])) for c, pts in zip(self.classes_, class_prototypes)])
                for (c, d_new), detector in zip(new_distances.items(), self.cdd):
                    detector.add_element(d_new)
                    if detector.detected_change():
                        self.drift_detected = True
            except Exception:
                print("Warning: Current Batch does not contain all labels!")
                # ValueError('zero-size array to reduction operation maximum which has no identity',)
                # In this batch not every label is present
        else:
            if not self.init_drift_detection:
                for elem, detector in zip(X.T, self.cdd):
                    for e in elem:
                        detector.add_element(e)
                        if detector.detected_change():
                            self.drift_detected = True

        return self.drift_detected

    def cd_handling(self, X, Y):
        #        print('cd handling')
        if self.replace:
            labels = np.concatenate([np.repeat(l, self.prototypes_per_class) for l in self.classes_])
            # new_prototypes = np.repeat(np.array([self.geometric_median(np.array([detector.window[-30:] for detector in self.cdd]).T)]),len(labels),axis=0)
            new_prototypes = np.array(
                [np.mean(np.array([detector.window[-30:] for detector in self.cdd]), axis=1) for l in labels])
            self.w_ = new_prototypes
            self.c_w_ = labels
            if type(self.initial_prototypes) == np.ndarray:
                self.initial_prototypes = np.append(new_prototypes, labels[:, None], axis=1)
        else:
            labels = self.classes_
            new_prototypes = np.array([self.geometric_median(X[Y == l]) for l in labels])
            self.w_ = np.append(self.w_, new_prototypes, axis=0)
            self.c_w_ = np.append(self.c_w_, labels, axis=0)
            self.prototypes_per_class = self.prototypes_per_class + 1
            if type(self.initial_prototypes) == np.ndarray:
                self.initial_prototypes = np.append(self.initial_prototypes,
                                                    np.append(new_prototypes, labels[:, None], axis=1), axis=0)

    def geometric_median(self, points):
        """
    Calculates the geometric median of an array of points.
    'minimize' -- scipy.optimize the sum of distances
        """

        points = np.asarray(points)

        if len(points.shape) == 1:
            # geometric_median((0, 0)) has too much potential for error.
            # Did the user intend a single 2D point or two scalars?
            # Use np.median if you meant the latter.
            raise ValueError("Expected 2D array")

        # objective function
        def aggregate_distance(x):
            return cdist([x], points).sum()

        # initial guess: centroid
        centroid = points.mean(axis=0)

        optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')

        return optimize_result.x


if __name__ == "__main__":
    
    s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
    s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)

    """1. Create stream"""
    stream = ReoccuringDriftStream(stream=s1,
                            drift_stream=s2,
                            random_state=None,
                            alpha=90.0, # angle of change grade 0 - 90
                            position=2000,
                            width=1)
    stream.prepare_for_use()
    rrslvq = ARSLVQ(prototypes_per_class=4, drift_detector="KS",sigma=2,confidence=0.1)

    model_names = ["rrslvq"]

    evaluator = EvaluatePrequential(show_plot=False,max_samples=5000,
    restart_stream=True,batch_size=50,metrics=['kappa', 'kappa_m', 'accuracy'])

    evaluator.evaluate(stream=stream, model=rrslvq,model_names=model_names)
