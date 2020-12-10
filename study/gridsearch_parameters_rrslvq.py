
from bix.evaluation.study import Study
from rrslvq import ReactiveRobustSoftLearningVectorQuantization
import itertools
import numpy as np
import pandas as pd
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


#Study Parameters
parallel = -1
study_size = 1000000
metrics = ['accuracy','kappa']
n_parameters = 60



cwd  = os.getcwd()
study = Study()
streams =  study.init_esann_si_streams()
os.chdir(cwd)
grid = {
"stat_size" : np.array([30]),
"gamma" : np.array([0.9,0.99]),
"sigma" : np.array([1,2,5,7,9]),
"prototypes_per_class" : np.array([1,2,5,7,9]),
"confidence" : np.array([0.0001])}


matrix = list(itertools.product(*[list(v) for v in grid.values()]))
if len(matrix) > 60:
    random_search = np.random.choice(len(matrix),size=n_parameters,replace=False)
    matrix = [matrix[i] for i in random_search]

best = []
iterations = len(matrix) * len(streams)

for i,stream in enumerate(streams):
    results = []
    print("Stream "+str(i+1)+" of "+ str(len(streams)))
    results.append(Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(param,stream,study_size) for param in matrix))
    results = np.array(results[0])
    best.append(results[np.argmax(results[:,-1])])
    np.savetxt("Gridsearch_"+str(stream.name)+".csv", results, delimiter=",")

names = [stream.name for stream in streams]
data = names.append(best[0].tolist())

df = pd.DataFrame(data)
df.to_csv("Summary Grid Search.csv")
