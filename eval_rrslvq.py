
from bix.evaluation.study import Study
from bix.classifiers.arslvq import ARSLVQ
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
    return [stream.name ]\
           +(evaluator._data_buffer.get_data(metric_id="accuracy", data_id="mean"))\
            +(evaluator._data_buffer.get_data(metric_id="kappa", data_id="mean"))



#Study Parameters
parallel = 18
study_size = 1000000
metrics = ['accuracy','kappa']


cwd  = os.getcwd()
study = Study()
streams =  study.init_esann_si_streams()
os.chdir(cwd)

parameters = pd.read_csv("Summary_Grid_Search.csv",index_col=0,header=0)
parameters = parameters.values[:,1:-1]

results = []

results.append(Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(param,stream,study_size)  for stream,param in zip(streams,parameters)))

results = results[0]
df = pd.DataFrame(results)

df_syn = df.drop([0,1,2,3,4,5])
df_real = df.iloc[[0,1,2,3,4,5],:]
df = df.append(["mean",df.mean()])
df_syn = df_syn.append(["mean",df_syn.mean()])
df_real = df_real.append(["mean",df_real.mean()])
df.to_csv("Summary_results_overall.csv",index=False)
df_syn.to_csv("Summary_results_syn.csv",index=False)
df_real.to_csv("Summary_results_real.csv",index=False)
