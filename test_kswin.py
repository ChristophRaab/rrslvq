import numpy as np
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.drift_detection import KSWIN
import numpy as np

# Imports
import numpy as np
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.drift_detection import KSWIN
import numpy as np
# Initialize KSWIN and a data stream
kswin = KSWIN(alpha=0.0001)
stream = SEAGenerator(classification_function = 2,\
    random_state = 112, balance_classes = False,noise_percentage = 0.28)
# Store detections 
detections = []
# Process stream via KSWIN and print detections 
print("\n--------------------\n")
for i in range(1000):
        data = stream.next_sample(10)
        batch = data[0][0][0]
        kswin.add_element(batch)
        if kswin.detected_change():
            print("\rIteration {}".format(i))
            print("\r KSWINReject Null Hyptheses")
            detections.append(i)
print("----- Number of detections: "+str(len(detections))+ " -----")

def test_kswin_init_alpha():
    """
    KSWIN alpha initalisiation test.
    alpha has range from (0,1)
    """
    try:
        KSWIN(alpha=-0.1)
    except ValueError:
        assert True
    else:
        assert False
    try:
        KSWIN(alpha=1.1)
    except ValueError:
        assert True
    else:
        assert False

    kswin = KSWIN(alpha=0.5)
    assert kswin.alpha == 0.5

def test_kswin_init_data():
    """
    KSWIN pre obtained data initalisiation test.
    data must be list
    """
    kswin = KSWIN(data="st")
    assert isinstance(kswin.window,list)

def test_kswin_window_size():
    """
    KSWIN window size initalisiation test.
    0 < stat_size <  w_size    
    """
    try:
        KSWIN(w_size=-10)
    except ValueError:
        assert True
    else:
        assert False
    try:
        KSWIN(w_size=10,stat_size=30)
    except ValueError:
        assert True
    else:
        assert False

def test_kswin_change_detection():
    """
    KSWIN change detector size initalisiation test. 
    At least 1 false positive must arisie due to the sensitive alpha, when testing the standard 
    Sea generator
    """
    kswin = KSWIN(alpha=0.001)
    stream = SEAGenerator(classification_function = 2,\
     random_state = 112, balance_classes = False,noise_percentage = 0.28)

    detections,mean = [],[]
    
    for i in range(1000):
        data = stream.next_sample(10)
        batch = data[0][0][0]
        mean.append(batch)
        kswin.add_element(batch)
        if kswin.detected_change():
            mean = []
            detections.append(i)
    assert len(detections) > 1

