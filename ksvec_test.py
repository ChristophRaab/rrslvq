from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
from cd_naive_bayes import cdnb

stream = ReoccuringDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                              drift_stream=SEAGenerator(random_state=112,
                                                        classification_function=2, noise_percentage=0.1),
                              alpha=90.0,
                              random_state=None,
                              position=2000,
                              width=100)


# stream = SEAGenerator()
start_size = 200
study_size = 20000
metrics  = ['accuracy','model_size',"running_time"]
# Initial training
stream.prepare_for_use()
stream.restart()


detectors = ["KSWIN", "ADWIN", "EDDM", "DDM", "KSVEC"]
cls = [cdnb(drift_detector=s) for s in detectors]

evaluator = EvaluatePrequential(show_plot=True, batch_size=10, max_samples=study_size, metrics=metrics,
                                output_file=stream.name + " " + str(detectors) + ".csv")

evaluator.evaluate(stream=stream, model=cls, model_names=detectors)

