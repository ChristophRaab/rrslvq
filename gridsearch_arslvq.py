
from bix.evaluation.study import Study
from bix.classifiers.arslvq import ARSLVQ
import itertools
import numpy as np
from skmultiflow.evaluation import EvaluatePrequential
from joblib import Parallel, delayed
import os
def evaluate(params,stream,study_size,metrics=['accuracy','kappa']):
    clf = ARSLVQ(gamma=params[1],sigma=params[2],prototypes_per_class=int(params[3]),confidence=params[4])
    stream.prepare_for_use()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics)

    model = evaluator.evaluate(stream=stream, model=clf)

    print(evaluator.get_mean_measurements())
    return list(params)+(evaluator._data_buffer.get_data(metric_id="accuracy", data_id="mean"))

cwd  = os.getcwd()
parallel = 10
study_size = 100000
metrics = ['accuracy','kappa']


study = Study()
streams =  study.init_esann_si_streams()
os.chdir(cwd)
streams = streams[6:]
grid = {
"stat_size" : np.array([10,30,50]),
"gamma" : np.array([0.7,0.9,0.99]),
"sigma" : np.arange(1, 11, 1),
"prototypes_per_class" : np.arange(1, 11, 1),
"confidence" : np.array([0.0001,0.001,0.01])}


matrix = list(itertools.product(*[list(v) for v in grid.values()]))
random_search = np.random.choice(len(matrix),size=90,replace=False)
matrix = [matrix[i] for i in random_search]

best = []
iterations = len(matrix) * len(streams)

for i,stream in enumerate(streams):
    results = []
    print("Stream "+str(i+1)+" of "+ str(len(streams)))
    results.append(Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(param,stream,study_size) for param in matrix))
    results = np.array(results)
    best.append(results[np.argmax(results[:,-1])])
    np.savetxt("Gridsearch_results_"+str(stream.name), results[0], delimiter=",")

np.savetxt("Summary Grid Search",best[0],delimiter=",")
# Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)
#
# streams  = s.init_real_world()
# Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)
# #
