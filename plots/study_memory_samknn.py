
from __future__ import division

from joblib import Parallel, delayed
from reoccuring_drift_stream import ReoccuringDriftStream
from bix.classifiers.rslvq import RSLVQ
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.lazy import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from rrslvq import ReactiveRobustSoftLearningVectorQuantization
from reoccuring_drift_stream import ReoccuringDriftStream
from bix.classifiers.rrslvq import RRSLVQ
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from bix.evaluation.study import Study
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
def init_classifiers():
    n_prototypes_per_class = 4
    sigma = 4
    rslvq = RSLVQ(prototypes_per_class=4, sigma=4)
    arslvq = ARSLVQ(prototypes_per_class=n_prototypes_per_class, sigma=sigma, confidence=0.0001, window_size=300)

    oza = OzaBaggingAdwin(base_estimator=KNN())
    adf = AdaptiveRandomForest()
    samknn = SAMKNN()
    hat = HAT()

    clfs = [samknn]
    names = ["SamKnn"]
    # clfs = [rslvq]
    # names = ["rslvq"]
    return clfs,names

def evaluate(stream,metrics,study_size):
    clfs,names = init_classifiers()
    stream.prepare_for_use()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics,
                                    output_file=stream.name+"_memory_other.csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel =2
study_size = 50000 #100000
metrics = ['accuracy','model_size']

s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)
mixed_ra = ReoccuringDriftStream(stream=s1, drift_stream=s2,random_state=None,alpha=90.0, position=2000,width=100,pause = 1000)
mixed_a = ConceptDriftStream(stream=s1,
                           drift_stream=s2,
                           alpha=90.0,
                           random_state=None,
                           position=int(study_size/2),
                           width=1)
sea_a = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                           drift_stream=SEAGenerator(random_state=112,
                                                     classification_function=2, noise_percentage=0.1),
                           alpha=90.0,
                           random_state=None,
                           position=int(study_size/2),
                           width=1)

sea_ra = ReoccuringDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                              drift_stream=SEAGenerator(random_state=112,
                                                        classification_function=2, noise_percentage=0.1),
                              alpha=90.0,
                              random_state=None,
                              position=2000,
                              width=1)

# metrics = ["accuracy","model_size"]
#evaluate(stream,clfs,metrics,names,study_size)
streams = s.init_standard_streams()  + s.init_reoccuring_standard_streams()
streams = [mixed_a,mixed_ra]
# for stream in streams:
#     evaluate(stream,metrics,study_size)
# for stream in streams:
#     evaluate(stream,metrics,study_size)

Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)
# streams  = s.init_real_world()
# Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)
# #
