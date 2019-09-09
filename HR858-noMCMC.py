import stability_functions as sf
import numpy as np
import os
import sys

mb, mc, md = sf.draw_masses()
num = 50
system = "HR858"
dir_path = os.path.dirname(os.path.realpath(__file__)) #directory of this program
out_dir = dir_path+"/output/"+system+"/"
os.system('mkdir %s'%out_dir)

for i in range(num):
    sim = sf.build_HR858(mb, mc, md)
    sim_num = (int(sys.argv[1]) - 1) * num + i
    name = system+"_start_"+str(sim_num)

    sf.replace_snapshot(sim, out_dir+name+'.bin')

    score = np.array([sim_num, sf.stability_score(sim)])

    if os.path.isfile("scores.npy"):
        scores = np.load("scores.npy")
    else:
        scores = np.empty(0)

    scores = np.append(scores, score)
    np.save("scores.npy", scores)