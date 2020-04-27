# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist
from kswin import KSWIN
from skmultiflow.drift_detection import ADWIN


class ReactiveRobustSoftLearningVectorQuantization(ClassifierMixin, BaseEstimator):
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
        If KS, use of Kolmogorov Smirnov test [1]_.
        IF DIST, monitoring class distances to detect outlier.
        IF ADWIN, use ADWIN detector [1]_.
    confidence : float, p-Value of Kolmogorov–Smirnov test(default=0.05)
    gamma : float, Decay Rate for Adadelta (default=0.9)
    replace : bool, True, replaces the current set of prototypes if concept
        drift is detected(default=0.05) and False adds a one prototype per class
        to the prototype set for representing the new concept
    windoprototype_setsize: float (default=100)
        Size of the sliding window for the KSWIN drift detector
    stat_size: float (default=30)
        Size of the statistic window for the KSWIN drift detector

    Notes
    -----
    RSSLVQ (Reactive Robust Soft Learning Vector Quantization) [1]_ is concept
    drift stream classifier, equiped with the KSWIN drift detector and the
    momentum based gradient descent to adapt fast to conceptual changes after
    detection. See documentation for KSWIN in the imported file.

    Attributes
    ----------
    prototypes : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    prototypes_classes : array-like, shape = [n_prototypes]
        Prototypes classes
    class_labels : array-like, shape = [n_classes]
        Array containing labels.

    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive
       Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,
    .. [2] Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
       In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448.
       Society for Industrial and Applied Mathematics, 2007.
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, random_state=112, drift_detector="KS", confidence=0.05,
                 gamma: float = 0.9, replace: bool = True, windoprototype_setsize=100, stat_size=30,):
        self.sigma = sigma

        self.random_state = random_state
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.class_labels  = []

        #### Reactive extensions  ####
        self.confidence = confidence
        self.counter = 0
        self.cd_detects = []
        self.drift_detector = drift_detector
        self.drift_detected = False
        self.replace = replace
        self.init_drift_detection = True
        self.windoprototype_setsize = windoprototype_setsize
        self.stat_size = stat_size
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
        nb_prototypes = self.prototypes_classes.size
        prototypes = self.prototype_set.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = x[i]
            c_xi = y[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])

                if self.prototypes_classes[j] == c_xi:
                    gradient = (self._p(j, xi, prototypes=self.prototype_set, y=c_xi) -
                                self._p(j, xi, prototypes=self.prototype_set)) * d
                else:
                    gradient = - self._p(j, xi, prototypes=self.prototype_set) * d

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
                self.prototype_set[j] += step
            #            """Implementation of Stochastical Gradient Descent"""

    #            n_data, n_dim = X.shape
    #            nb_prototypes = self.prototypes_classes.size
    #            prototypes = self.prototype_set.reshape(nb_prototypes, n_dim)
    #
    #            for i in range(n_data):
    #                xi = X[i]
    #                c_xi = y[i]
    #                for j in range(prototypes.shape[0]):
    #                    d = (xi - prototypes[j])
    #                    c = 1/ self.sigma
    #                    if self.prototypes_classes[j] == c_xi:
    #                        # Attract prototype to data point
    #                        self.prototype_set[j] += c * (self._p(j, xi, prototypes=self.prototype_set, y=c_xi) -
    #                                     self._p(j, xi, prototypes=self.prototype_set)) * d
    #                    else:
    #                        # Distance prototype from data point
    #                        self.prototype_set[j] -= c * self._p(j, xi, prototypes=self.prototype_set) * d

    def _costf(self, x, w, **kwargs):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(d)
        return -d / (2 * self.sigma)

    def _p(self, j, e, y=None, prototypes=None, **kwargs):
        if prototypes is None:
            prototypes = self.prototype_set
        if y is None:
            fs = [self._costf(e, w, **kwargs) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i], **kwargs) for i in
                  range(prototypes.shape[0]) if
                  self.prototypes_classes[i] == y]

        fs_max = max(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        o = np.math.exp(
            self._costf(e, prototypes[j], **kwargs) - fs_max) / s
        return o

    def get_prototypes(self):
        """Returns the prototypes"""
        return self.prototype_set

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
        return np.array([self.prototypes_classes[np.array([self._costf(xi, p) for p in self.prototype_set]).argmax()] for xi in x])

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
        check_is_fitted(self, ['prototype_set', 'prototypes_classes'])
        x = validation.column_or_1d(x)
        if y not in self.class_labels :
            raise ValueError('y must be one of the labels\n'
                             'y=%s\n'
                             'labels=%s' % (y, self.class_labels ))
        s1 = sum([self._costf(x, self.prototype_set[i]) for i in
                  range(self.prototype_set.shape[0]) if
                  self.prototypes_classes[i] == y])
        s2 = sum([self._costf(x, w) for w in self.prototype_set])
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
                self.class_labels  = np.asarray(classes)
                self.protos_initialized = np.zeros(self.class_labels .size)
            else:
                self.class_labels  = unique_labels(train_lab)
                self.protos_initialized = np.zeros(self.class_labels .size)

        nb_classes = len(self.class_labels )
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
                self.prototype_set = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
                self.prototypes_classes = np.empty([nb_ppc.sum()], dtype=self.class_labels .dtype)
            pos = 0
            for actClassIdx in range(len(self.class_labels )):
                actClass = self.class_labels [actClassIdx]
                nb_prot = nb_ppc[actClassIdx]  # nb_ppc: prototypes per class
                if (self.protos_initialized[actClassIdx] == 0 and actClass in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == actClass, :], 0)
                    self.prototype_set[pos:pos + nb_prot] = mean + (
                            random_state.rand(nb_prot, nb_features) * 2 - 1)
                    if math.isnan(self.prototype_set[pos, 0]):
                        print('Prototype is NaN: ', actClass)
                        self.protos_initialized[actClassIdx] = 0
                    else:
                        self.protos_initialized[actClassIdx] = 1

                    self.prototypes_classes[pos:pos + nb_prot] = actClass
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.prototype_set = x[:, :-1]
            self.prototypes_classes = x[:, -1]
            if self.prototype_set.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.prototype_set.shape[0], self.prototype_set.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.prototypes_classes) != set(self.class_labels ):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.class_labels , self.prototypes_classes))
        if self.initial_fit:
            # Next two lines are Init for Adadelta/RMSprop
            self.squared_mean_gradient = np.zeros_like(self.prototype_set)
            self.squared_mean_step = np.zeros_like(self.prototype_set)
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

        if set(unique_labels(y)).issubset(set(self.class_labels )) or self.initial_fit == True:
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
        pd.DataFrame(self.prototype_set).to_csv("Prototypes.csv")
        pd.DataFrame(self.prototypes_classes).to_csv("Prototype_Labels.csv")
        pd.DataFrame(X).to_csv("Data.csv")
        pd.DataFrame(y).to_csv("Labels.csv")
        self._optimize(X, y, random_state)
        pd.DataFrame(self.prototype_set).to_csv("Prototypes1.csv")
        pd.DataFrame(self.prototypes_classes).to_csv("Prototype_Labels1.csv")

    def calcDistances(self, pts, x):
        dists = []
        for p in pts:
            for elem in x:
                dists.append(np.linalg.norm(p - elem))
        return np.max(dists)

    def concept_drift_detection(self, X, Y):
        if self.init_drift_detection:
            if self.drift_detector == "KS":
                self.cdd = [KSWIN(alpha=self.confidence, prototype_setsize=self.windoprototype_setsize, stat_size=self.stat_size) for elem in
                            X.T]
            if self.drift_detector == "ADWIN":
                self.cdd = [ADWIN(delta=self.confidence) for elem in X.T]
            if self.drift_detector == "DIST":
                self.cdd = [KSWIN(self.confidence, prototype_setsize=self.windoprototype_setsize) for c in self.class_labels ]
        self.init_drift_detection = False
        self.drift_detected = False

        if self.drift_detector == "DIST":
            try:
                class_prototypes = [self.prototype_set[self.prototypes_classes == elem] for elem in self.class_labels ]
                neprototype_setdistances = dict(
                    [(c, self.calcDistances(pts, X[Y == c])) for c, pts in zip(self.class_labels , class_prototypes)])
                for (c, d_new), detector in zip(neprototype_setdistances.items(), self.cdd):
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
            labels = np.concatenate([np.repeat(l, self.prototypes_per_class) for l in self.class_labels ])
            # neprototype_setprototypes = np.repeat(np.array([self.geometric_median(np.array([detector.window[-30:] for detector in self.cdd]).T)]),len(labels),axis=0)
            neprototype_setprototypes = np.array(
                [np.mean(np.array([detector.window[-self.stat_size:] for detector in self.cdd]), axis=1) for l in labels])
            self.prototype_set = neprototype_setprototypes
            self.prototypes_classes = labels
            if type(self.initial_prototypes) == np.ndarray:
                self.initial_prototypes = np.append(neprototype_setprototypes, labels[:, None], axis=1)
        else:
            labels = self.class_labels
            neprototype_setprototypes = np.array([self.geometric_median(X[Y == l]) for l in labels])
            self.prototype_set = np.append(self.prototype_set, neprototype_setprototypes, axis=0)
            self.prototypes_classes = np.append(self.prototypes_classes, labels, axis=0)
            self.prototypes_per_class = self.prototypes_per_class + 1
            if type(self.initial_prototypes) == np.ndarray:
                self.initial_prototypes = np.append(self.initial_prototypes,
                                                    np.append(neprototype_setprototypes, labels[:, None], axis=1), axis=0)

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


