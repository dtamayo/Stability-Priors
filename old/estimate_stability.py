import stability_functions as sf
import numpy as np
import sys
import pandas as pd
import dask.dataframe as dd
from multiprocessing import Pool
import os

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
system = systems[int(sys.argv[1])]
sf.valid_system(system)
sim_names = system + "/" + system
n = int(sys.argv[2])
# n_workers = os.cpu_count()
n_workers = 8
print("found " + str(n_workers) + " workers")


def pred(nsim):
    return sf.pred(sim_names, nsim)

def get_k(row):
    return sf.get_k(sim_names, row)

pool = Pool(processes=n_workers, initializer=sf.init_process)

nsim_list = np.arange(0, n)
res = pool.map(pred, nsim_list)

df1 = sf.add_k_cols(pd.DataFrame(nsim_list, columns=['sim']))
df1['probstability'] = res

dasklabels = dd.from_pandas(df1, npartitions=n_workers)
df = dasklabels.apply(get_k, axis=1, meta=df1).compute(scheduler='processes')

df["m1"] *= sf.earth_mass_p_solar_mass
df["m2"] *= sf.earth_mass_p_solar_mass
df["m3"] *= sf.earth_mass_p_solar_mass
df["m1+m2"] = df["m1"] + df["m2"]
df["m2+m3"] = df["m2"] + df["m3"]
df["total m"] = df["m1"] + df["m2"] + df["m3"]

df.to_csv(sim_names + ".csv")