import stability_functions as sf
import numpy as np
import os
import time
import sys
import rebound

id_ = int(sys.argv[1]) - 1

scores_file = np.load("scores.npy")

sim_nums = scores_file[::2].astype(int)
scores = scores_file[1::2]

target = 0.9

amount = 20
inds = np.argpartition(np.abs(scores - target), amount)
ind = inds[id_]
sim_num = sim_nums[ind]

system = "HR858"

dir_path = os.path.dirname(os.path.realpath(__file__)) #directory of this program
out_dir = dir_path+"/output/"+system+"/"
os.system('mkdir %s'%out_dir)

name = out_dir+system+"_start_"+str(sim_num)+".bin"

print("simulation number: " + str(sim_num))
print("recorded score: "+str(scores[ind]))

sim = rebound.SimulationArchive(name)[0]
score = sf.stability_score(sim)
print("new score: "+str(score))

maxorbs = float(1e9)
sim = rebound.SimulationArchive(name)[0]

sim.integrator = 'whfast'
sim.G = 1
sim.ri_whfast.safe_mode = 0
sim.collision = 'direct'
sim.collision_resolve = sf.collision

dt = 2.*np.sqrt(3)/100. #0.0346410162
P1 = sim.particles[1].P
sim.dt = dt*P1 # ~3% orbital period

tmax = maxorbs*P1

#save simulation archive
sim.automateSimulationArchive(out_dir+system+'_'+str(sim_nums[ind])+'_SA.bin', interval=tmax/1000, deletefile=True)

#simulate
E0 = sim.calculate_energy()
world_t0 = time.time()
sim_t0 = sim.t

print("starting simulation")
try:
    sim.integrate(tmax)
    unstable=0
except:
    unstable=1
print("finished simulation")

sf.replace_snapshot(sim, out_dir+system+'_'+str(sim_nums[ind])+'_final.bin')

Ef = sim.calculate_energy()
Eerr = abs((Ef-E0)/E0)

orbits = (sim.t - sim_t0) / P1

print("%.0f orbits completed"%(orbits))
print("%.2f%% of max_orbits"%(100 * orbits /  maxorbs))
print("took %.0f s"%(time.time() - world_t0))
print("%.4f%% energy error"%(Eerr*100))
