from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
from scipy import stats

class KSWIN(BaseDriftDetector):
    """ Kolmogorov-Smirnov Windowing method for concept drift detection.

    Parameters
    ----------
    alpha: float (default=0.005)
        Probability for the test statistic of the Kolmogorov-Smirnov-Test
        The alpha parameter is very sensitive, therefore should be set
        below 0.01.

    w_size: float (default=100)
        Size of stored samples in the sliding window

    stat_size: float (default=30)
        Size of the statistic window

    data: list (default=None)

     Notes
    -----
    KSWIN (Kolmogorov-Smirnov Windowing) [1]_ is a concept change detection 
    method based on the Kolmogorov-Smirnov (KS) statistical test. KS-test 
    is a statistical test with no assumption of underlying the data
    distribution. KSWIN is able to monitor data or perfomance distributions.

    KSWIN maintaines a sliding window `\Psi`  of fixed size `n` (w_size). 
    The last `r` (stat_size) samples of `\Psi` are assumed to represent the
    last concept `R`. From the frist `n-r` samples of `\Psi`, `r` sampels are
    uniformly drawn, which represent an approximated last concept `W`. 
    last concept. 

    The KS-test is performed on the windows `R` and `W` of same size. KS-test
    compares the distance of the empirical cumulative data distribution `dist(R,W)`.

    A concept drift is detected by KSWIN if:
    
    * `dist(R,W) > \sqrt{-\frac{ln\alpha}{r}}` 

    -> The difference in empirical data distributions between the windows `R`
    and `W` are to large to be stationary. 

    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive 
       Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,

    Examples
    ----------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> from skmultiflow.drift_detection import KSWIN
    >>> import numpy as np
    >>> # Initialize KSWIN and a data stream
    >>> kswin = KSWIN(alpha=0.0001)
    >>> stream = SEAGenerator(classification_function = 2,\
    >>>     random_state = 112, balance_classes = False,noise_percentage = 0.28)
    >>> # Store detections 
    >>> detections = []
    >>> # Process stream via KSWIN and print detections 
    >>> print("\n--------------------\n")
    >>> for i in range(1000):
    >>>         data = stream.next_sample(10)
    >>>         batch = data[0][0][0]
    >>>         kswin.add_element(batch)
    >>>         if kswin.detected_change():
    >>>             print("\rIteration {}".format(i))
    >>>             print("\r KSWINReject Null Hyptheses")
    >>>             detections.append(i)
    >>> print("----- Number of detections: "+str(len(detections))+ " -----")
        """
    def __init__(self, alpha=0.005, w_size=100, stat_size=30, data=None):

        self.w_size = w_size
        self.stat_size = stat_size 
        self.alpha = alpha
        self.change_detected = False
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.w_size < 0:
            raise ValueError("w_size must be greater than 0")

        if self.w_size < self.stat_size:
            raise ValueError("stat_size must be smaller than w_size")

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
        self.change_detected = False
        pass
    
    def get_info(self):
         return "KSwin Change: Current P-Value "+str(self.p_value)
