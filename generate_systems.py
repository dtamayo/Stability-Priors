import stability_functions as sf
import sys

systems = ["HR858", "K431", "TOI270", "L98-59"]
system = systems[int(sys.argv[1])]
sf.valid_system(system)
sim_names = system + "/" + system
n = int(sys.argv[2])

mb, mc, md = sf.draw_masses(system, int(1e4))

for i in range(n):
    name = sim_names + "_sa_%d.bin"%i
    sf.replace_snapshot(sf.build_forecasted_system(system, mb, mc, md), name)