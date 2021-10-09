import re
import glob
import numpy as np
import pandas as pd

for fname in list(glob.glob("AoS/data/times-*runs=5*.csv")):
    df = pd.read_csv(fname)
    df.drop(['id'],axis=1,inplace=True)
    df["Total Time"] = df.sum(axis=1)
    print(fname.split("(")[1][:-3],":")
    for col in df.columns:
        print("    {} : {:.4g} +- {:.4g}".format(col,df[col].mean(),df[col].std()))
    del df

print("\n")

for fname in list(glob.glob("SoA/data/times-*runs=5*.csv")):
    df = pd.read_csv(fname)
    df.drop(['id'],axis=1,inplace=True)
    df["Total Time"] = df.sum(axis=1)
    print(fname.split("(")[1][:-3],":")
    for col in df.columns:
        print("    {} : {:.4g} +- {:.4g}".format(col,df[col].mean(),df[col].std()))
    del df