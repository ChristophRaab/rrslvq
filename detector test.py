import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from cd_naive_bayes import cdnb
from bix.evaluation.study import Study
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Test settings
n_batches = 100000
batch_size = 10
start_size = 200
study_size = 1

# Ground truth labels of non-reoccurring concept drift streams
cd_truth = np.zeros(n_batches)
cd_truth[int(n_batches/2)] = 1

# Ground truth label of reoccurring concept drift streams
rec_truth = np.zeros(n_batches)
for idx in range(rec_truth.size):
    rec_truth[idx] = 1 if (idx*batch_size) % 1000  == 0 and idx != 0 else 0



# Accuracy placeholder
acc = [[] for i in range(study_size)]

# Initialization of streams
cwd  = os.getcwd()
s = Study()
s_streams = s.init_standard_streams()
r_streams = s.init_reoccuring_standard_streams()
cd_truth = np.concatenate([np.tile(cd_truth,len(s_streams)),np.tile(rec_truth,len(r_streams))])
os.chdir(cwd)

# Setting Concept Drift position for non-reoccurring concept drift streams
for s in s_streams:
    s.position = int(n_batches*batch_size/2)

# Setting frequency of reoccurring concept drift streams
for s in r_streams:
    s.position = 1000

# Merge of stream array
streams = s_streams+r_streams



# Detectors with Naive Bayes classifier
# plus concept drift detection placeholder
detectors = ["KSWIN", "ADWIN", "EDDM", "DDM"]
cls = [cdnb(drift_detector=s) for s in detectors]
cd_pred = np.zeros((len(detectors), study_size, len(streams),n_batches))

# Testscript
for i in range(study_size):

    for j,stream in enumerate(streams):
        print(stream.name + "\n")

        # Initial training
        stream.prepare_for_use()
        stream.restart()
        X,y = stream.next_sample(start_size)


        for c in cls:
            c.partial_fit(X, y)

        # Prediction accuracy placeholder
        label_pred = [[] for c in cls]
        label_truth = []

        for b in range(n_batches):

            X, y = stream.next_sample(batch_size)
            label_truth.extend(y)

            # Training, detection and prediction
            for idx,c in enumerate(cls):
                y_pred = c.predict(X)
                label_pred[idx].extend(y_pred)
                c.partial_fit(X,y)
                if c.drift_detected == True:
                    cd_pred[idx][i][j][b] = 1

        # Merge results
        label_pred = np.array(label_pred)
        label_truth = np.array(label_truth)

        for pred in label_pred:
            c_acc = (label_truth == pred).sum()/label_truth.size
            acc[i].append(c_acc)


# Accuracy: Merge of results
acc = np.array(acc).reshape((study_size,len(streams),len(detectors)))
mean_c = np.mean(acc,axis=(0))
mean = np.mean(acc, axis=(0,1)).round(2).astype(float)
df = pd.DataFrame(list(mean_c)+list([mean]),columns=detectors,index=[stream.name for stream in streams]+["Mean"])
df.to_csv("prediction_results.csv")

# Confusion Matrix: Calculation and plot
result = []
for i in range(len(detectors)):
    c_matrix = confusion_matrix(cd_truth,cd_pred[i][0].reshape(cd_truth.shape))
    fig, ax = plt.subplots()
    sns.heatmap(c_matrix, annot=True, ax=ax,cmap="Blues");  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted Class');
    ax.set_ylabel('True Class');
    plt.show()
    fig.savefig("confusion_matrix_"+detectors[i]+".eps",edpi=1000, format='eps',quality=95)

    result.append(list(c_matrix.flatten()))
df = pd.DataFrame(result)
df.to_csv("confusion_matrix.csv",index=None,header=["True Negative","False Positive","False Negative","True Positive"])
