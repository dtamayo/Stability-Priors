import stability_functions as sf
import numpy as np
import os

mb, mc, md = sf.draw_masses()

for i in range(50):
    sim, Ps, ms, es, incs = sf.build_HR858(mb, mc, md, return_extra=True)
    score = sf.stability_score(sim)

    if os.path.isfile("scores.npy") and os.path.isfile("all_es.npy") and os.path.isfile("all_ms.npy") and os.path.isfile("all_Ps.npy") and os.path.isfile("all_incs.npy"):
        scores = np.load("scores.npy")
        all_es = np.load("all_es.npy")
        all_ms = np.load("all_ms.npy")
        all_Ps = np.load("all_Ps.npy")
        all_incs = np.load("all_incs.npy")
    else:
        scores = np.empty(0)
        all_es = np.empty(0)
        all_ms = np.empty(0)
        all_Ps = np.empty(0)
        all_incs = np.empty(0)

    scores = np.append(scores, score)
    all_es = np.append(all_es, es)
    all_ms = np.append(all_ms, ms)
    all_Ps = np.append(all_Ps, Ps)
    all_incs = np.append(all_incs, incs)
    np.save("scores.npy", scores)
    np.save("all_es.npy", all_es)
    np.save("all_ms.npy", all_ms)
    np.save("all_Ps.npy", all_Ps)
    np.save("all_incs.npy", all_incs)