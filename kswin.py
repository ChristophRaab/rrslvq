from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
from scipy import stats


class KSWIN(BaseDriftDetector):
    """KSWIN - Concept Drift Detector
    """

    def __init__(self, alpha=0.05, w_size=100, stat_size=30, data=None):

        self.w_size = w_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.change_detected = False;
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        if type(data) != list or type(data) == None:
            self.window = []
        else:
            self.window = data
        pass

    def add_element(self, value):
        self.n += 1
        currentLength = len(self.window)
        if currentLength >= self.w_size:
            self.window.pop(0)
            rnd_window = np.random.choice(self.window[:-self.stat_size], self.stat_size)
            # rnd_window = self.window[:-self.stat_size]
            (st, self.p_value) = stats.ks_2samp(rnd_window, self.window[-self.stat_size:])

            if self.p_value <= self.alpha and 0.1 < st:
                self.change_detected = True
                self.window = self.window[-self.stat_size:]
            else:
                self.change_detected = False
        else:
            self.change_detected = False

        self.window.insert(currentLength, value)
        pass

    def detected_change(self):
        return self.change_detected

    def reset(self):
        self.alpha = 0
        self.window = []
        self.change_detected = False;
        pass

    def get_info(self):
        return "KSwin Change: Current P-Value " + str(self.p_value)
