from sklearn.base import BaseEstimator, ClassifierMixin
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from bix.detectors.kswin import KSWIN
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import  EDDM
from skmultiflow.drift_detection.ddm import DDM

class cdht(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=0.001,drift_detector="KSWIN"):
        self.classifier = HoeffdingTree()
        self.init_drift_detection = True
        self.drift_detector = drift_detector.upper()
        self.confidence = alpha
        self.n_detections = 0

    def partial_fit(self, X, y, classes=None):
            """
            Calls the MultinomialNB partial_fit from sklearn.
            ----------
            x : array-like, shape = [n_samples, n_features]
              Training vector, where n_samples in the number of samples and
              n_features is the number of features.
            y : array, shape = [n_samples]
              Target values (integers in classification, real numbers in
              regression)
            Returns
            --------
            """
            if self.concept_drift_detection(X, y):
                self.classifier.reset()

            self.classifier.partial_fit(X, y,classes)
            return self

    def predict(self, X):
        return self.classifier.predict(X)

    def concept_drift_detection(self, X, Y):
        if self.init_drift_detection:
            if self.drift_detector == "KSWIN":
                self.cdd = [KSWIN(w_size = 100, stat_size = 30, alpha=self.confidence) for elem in X.T]
            if self.drift_detector == "ADWIN":
                self.cdd = [ADWIN() for elem in X.T]
            if self.drift_detector == "DDM":
                self.cdd = [DDM() for elem in X.T]
            if self.drift_detector == "EDDM":
                self.cdd = [EDDM() for elem in X.T]
            self.init_drift_detection = False
        self.drift_detected = False

        if not self.init_drift_detection:
            for elem, detector in zip(X.T, self.cdd):
                for e in elem:
                    detector.add_element(e)
                    if detector.detected_change():
                        self.drift_detected = True
                        self.n_detections = self.n_detections +1

        return self.drift_detected


# if name=="__main__":
#     from skmultiflow import