# This demo file compares the OzaBagginAdwin algorithms with the ReactiveRobustSoftLearningVectorQuantization
# on Mixed Generator with high rates of drift.


from skmultiflow.data import MIXEDGenerator
from study.reoccuring_drift_stream import ReoccuringDriftStream
from rrslvq import ReactiveRobustSoftLearningVectorQuantization
from skmultiflow.meta import OzaBaggingAdwin
from skmultiflow.lazy import KNN
from skmultiflow.evaluation import EvaluatePrequential

s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)


stream = ReoccuringDriftStream(stream=s1, drift_stream=s2,random_state=None,alpha=90.0, position=2000,width=1)


rrslvq = ReactiveRobustSoftLearningVectorQuantization(prototypes_per_class=4,sigma=12)
oza = OzaBaggingAdwin(base_estimator=KNN(), n_estimators=2)



evaluator = EvaluatePrequential(show_plot=True,max_samples=10000,
restart_stream=True,batch_size=10,metrics=[ 'accuracy', 'kappa', 'kappa_m'])

evaluator.evaluate(stream=stream, model=[oza, rrslvq],model_names=["oza","rrslvq"])