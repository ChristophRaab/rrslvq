import glob
import pandas as pd
import numpy as np



files = glob.glob("*.csv")
print(len)
results =[]
for file in files:
    df = pd.read_csv(file,index_col=None,header=None,dtype=float)
    results.append([file.replace("Gridsearch_","").replace(".csv","")] +df.iloc[int(df[5].idxmax()),:].values.tolist())
df1 = pd.DataFrame(results)
len(df)

files = glob.glob("results_full/*.csv")

results =[]
for file in files:
    df = pd.read_csv(file,index_col=None,header=None,dtype=float)
    results.append([file.replace("Gridsearch_","").replace(".csv","")] +df.iloc[int(df[5].idxmax()),:].values.tolist())
df2 = pd.DataFrame(results)

results = []
for one,two in zip(df1.values,df2.values):

    if one[-1] < two[-1]:
        results.append(two)
    else:
        results.append(one)

df = pd.DataFrame(results)
df.to_csv("Summary_Grid_Search.csv")
df_syn = df.drop([0,1,2,8,9,17])
df_real = df.iloc[[0,1,2,8,9,17],:]
df = df.append(["mean",np.mean(df.iloc[:,-1])])
df_syn = df_syn.append(["mean",np.mean(df_syn.iloc[:,-1])])
df_real = df_real.append(["mean",np.mean(df_real.iloc[:,-1])])
df.to_csv("merge_results.csv",index=False)
df_syn.to_csv("merge_results_syn.csv",index=False)
df_real.to_csv("merge_results_real.csv",index=False)
