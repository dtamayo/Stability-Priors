import numpy as np
import pandas as pd
import sys
import rebound

def collision(reb_sim, col):
    reb_sim.contents._status = 5
    return 0

if len(sys.argv) > 3:
    shadow = bool(int(sys.argv[3]))
else:
    shadow = False

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
system = systems[int(sys.argv[1])]
sim_names = system + "/" + system

df = pd.read_csv(sim_names + ".csv", index_col=0)
df = df.sort_values("probstability", ascending=False)
nsim = int(df.iloc[int(sys.argv[2])]["sim"])

sim = rebound.SimulationArchive(sim_names + "_sa_%d.bin"%nsim)[0]

if shadow:
    df2 = pd.read_csv(sim_names + "_%d_orbits.csv"%nsim, index_col=0)
    if not df2["stable"][0]:
        sys.exit("not worth running a shadow system for dummy thin system")
    else:
        sim.particles[2].x *= 1 + 1.e-11
        
P1 = sim.particles[1].P
try:
    sim.integrate(1e9 * P1, exact_finish_time=0)
    orbits = sim.t / P1
    res = True
except:
    orbits = sim.t / P1
    res = False

if shadow:
    df2["orbits_shadow"] = orbits
    df2["stable_shadow"] = res
else:
    df2 = pd.DataFrame(data=[nsim], columns=["sim"])
    df2["orbits"] = orbits
    df2["stable"] = res
df2.to_csv(sim_names + "_%d_orbits.csv"%nsim)