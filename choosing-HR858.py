import stability_functions as sf
import numpy as np
import os
import time
import sys

mb, mc, md = sf.draw_masses()

id_ = int(sys.argv[1]) - 1

scores = np.load("scores.npy")
all_es = np.load("all_es.npy")
all_ms = np.load("all_ms.npy")
all_Ps = np.load("all_Ps.npy")
all_incs = np.load("all_incs.npy")

target = 0.8

amount = 10
inds = np.argpartition(np.abs(scores - target), amount)
# print(scores[inds[0:amount]])
ind = 3*inds[id_]

# ind = np.argmin(np.abs(scores - target))

# sorted_scores = np.sort(scores)
# ind = np.where(sorted_scores[-2]==scores)[0][0]

sim, Ps, ms, es, incs = sf.build_chosen_HR858(all_ms[ind:ind+3], all_es[ind:ind+3], Ps=all_Ps[ind:ind+3], incs=all_incs[ind:ind+3], return_extra=True)
score = sf.stability_score(sim)

print(score, Ps, ms, es, incs, sep="\n")

system = "HR858"

maxorbs = float(1e9)   # / 1e6
name = system+"_"+str(int(np.log10(maxorbs)))+"_"+str(id_)

sim.integrator = 'whfast'
sim.G = 1
sim.ri_whfast.safe_mode = 0
sim.collision = 'direct'
sim.collision_resolve = sf.collision

dt = 2.*np.sqrt(3)/100. #0.0346410162
P1 = sim.particles[1].P
sim.dt = dt*P1 # ~3% orbital period

#how long you are willing to simulate this system right now
tmax = maxorbs*P1

#save simulation archive
# dir_path = os.path.dirname(os.path.realpath(__file__)) #directory of this program
# out_dir = dir_path+"/output/"
# os.system('mkdir %s'%out_dir)
# sim.automateSimulationArchive(out_dir+'%s_SA.bin'%name, interval=tmax/1000, deletefile=True)

#simulate
E0 = sim.calculate_energy()
world_t0 = time.time()
sim_t0 = sim.t

#plot system now
# fig1 = rebound.OrbitPlot(sim, unitlabel="[AU]", color=True, periastron=True)

print("starting simulation")
try:
    sim.integrate(tmax)
    unstable=0
except:
    unstable=1
print("finished simulation")

# sf.replace_snapshot(sim, out_dir+'%s_final.bin'%name)

Ef = sim.calculate_energy()
Eerr = abs((Ef-E0)/E0)

#plot system after tmax or when things went to crap
# fig2 = rebound.OrbitPlot(sim, unitlabel="[AU]", color=True, periastron=True)

orbits = (sim.t - sim_t0) / P1
#print the amount of orbits that have occured
print("%.0f orbits completed"%(orbits))
print("%.2f%% of max_orbits"%(100 * orbits /  maxorbs))
print("took %.0f s"%(time.time() - world_t0))

# sim = sa.getSimulation(0)
# print(sim.t)

# sim = sa.getSimulation(sa.tmax, mode="close")
# print(sim.t)