import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
import pandas as pd
time = np.arange(0, 5000, 1)
position = 1000
width = 1000

probability_drift = np.array([1.0 / (1.0 + np.exp(-4.0 * float(t - position) / float(width))) for t in time])
inv_probability_drift = np.array([1 - p for p in probability_drift])

            

fig, ax = plt.subplots()

ax.plot(time,probability_drift)
ax.plot(time,inv_probability_drift)
plt.show()
