import glob
import numpy as np
import pandas as pd

file_refs = "AoS/data/cd3d(naive,simple,runs=1)-P(seed=123123123,N=100000,h=0.05)-B(Nx=10,Ny=10,Nz=10)-D(0,0,0,1,1,1).csv"
files_SoA = glob.glob("SoA/data/cd3d*.csv")
files_AoS = glob.glob("AoS/data/cd3d*.csv")

for file in files_AoS:
    df0 = pd.read_csv(file_refs)
    df1 = pd.read_csv(file)
    df1 = df1.sort_values(by="id").reset_index(drop=True)
    
    print(file)
    print("mean +- std: {0:.20g} +- {1:.20g} / max_abs : {2:.20g}\n".format(
                                                       (df1["rho"]-df0["rho"]).mean(),
                                                       (df1["rho"]-df0["rho"]).std(),
                                                       (df1["rho"]-df0["rho"]).abs().max()))
    
    del df0
    del df1
    
for file in files_SoA:
    df0 = pd.read_csv(file_refs)
    df1 = pd.read_csv(file)
    df1 = df1.sort_values(by="id").reset_index(drop=True)
    
    print(file)
    print("mean +- std: {0:.20g} +- {1:.20g} / max_abs : {2:.20g}\n".format(
                                                       (df1["rho"]-df0["rho"]).mean(),
                                                       (df1["rho"]-df0["rho"]).std(),
                                                       (df1["rho"]-df0["rho"]).abs().max()))
    
    del df0
    del df1