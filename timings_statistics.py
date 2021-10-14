#
# SPDX-License-Identifier:  BSD-3-Clause
#
# timings_statistics.py :
#     Print the statistics for the different 
#     computation modes
#
# (C) Copyright 2021 José Hugo Elsas
# Author: José Hugo Elsas <jhelsas@gmail.com>
#

import re
import glob
import numpy as np
import pandas as pd

for fname in list(glob.glob("AoS/data/times-*runs=*.csv")):
    df = pd.read_csv(fname)
    df.drop(['id'],axis=1,inplace=True)
    df["Total Time"] = df.sum(axis=1)
    print(fname.split("(")[1][:-3],":")
    for col in df.columns:
        print("    {} : {:.7g} +- {:.7g} : {:.4g}% +- {:.4g}%".format(col,df[col].mean(),df[col].std()
                                                                    ,100*df[col].mean()/df["Total Time"].mean(),
                                                                     100*df[col].std()/df["Total Time"].mean()
                                                                    ))
    del df

print("\n")

for fname in list(glob.glob("SoA/data/times-*runs=*.csv")):
    df = pd.read_csv(fname)
    df.drop(['id'],axis=1,inplace=True)
    df["Total Time"] = df.sum(axis=1)
    print(fname.split("(")[1][:-3],":")
    for col in df.columns:
        print("    {} : {:.7g} +- {:.7g}  : {:.4g}% +- {:.4g}%".format(col,df[col].mean(),df[col].std()
                                                                     ,100*df[col].mean()/df["Total Time"].mean(),
                                                                     100*df[col].std()/df["Total Time"].mean()
                                                                    ))
    del df