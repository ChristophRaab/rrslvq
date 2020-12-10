# Reactive Soft Prototype Computing for frequent reoccurring Concept Drift
Python source code of the publication *Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020*. 

The repository contains the Kolmogorov-Smirnov Windowing (KSWIN) concept drift detector and the Reactive Robust Soft Learning Vector Quantization (RRSLVQ), but the later is the main contribution of the paper.

The RRSLVQ is made for classifying streams with very high rates of drift while maintaining stabilit during active concept drift

## Demo of RRSLVQ and a demonstration of stability during drift is available at 
``demo.py``


Requirements:
scikit-multiflow:https://github.com/scikit-multiflow/scikit-multiflow
bix: https://github.com/ChristophRaab/bix
