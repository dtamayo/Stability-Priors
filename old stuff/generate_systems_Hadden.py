import stability_functions as sf
import sys

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
system = systems[int(sys.argv[1])]
sim_names = system + "/" + system
n = int(sys.argv[2])

for i in range(n):
    name = sim_names + "_logm_start_%d.bin"%i
    sf.replace_snapshot(sf.build_Hadden_system(system, logm=True), name)

for i in range(n):
    name = sim_names + "_loge_start_%d.bin"%i
    sf.replace_snapshot(sf.build_Hadden_system(system, loge=True), name)
