import stability_functions as sf
import numpy as np
import sys
import corner
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
matplotlib.use('Agg')

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
system = systems[int(sys.argv[1])]
sim_names_base = system + "/" + system

name_adds = ["loge", "logm"]
name_add = name_adds[int(sys.argv[2])]
sim_names = sim_names_base + "_" + name_add

df = pd.read_csv(sim_names + "_SPOCK.csv", index_col=0)
scores = 2 * df["probstability"]
n = len(scores)
# effective sample size using eq 9.13 https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf
effective_sample_size = n * np.mean(scores) ** 2 / np.mean(scores ** 2)
print(effective_sample_size)
print(n)
print(effective_sample_size / n)

plot_str = system + "_" + add_str

labels = ["e1","e2", "e3", "m1", "m2", "m3"]
labels2 = [r"$e_1$", r"$e_2$", r"$e_3$", r"$m_1$", r"$m_2$", r"$m_3$"]
fig = corner.corner(df[labels], labels=labels2, quantiles=[0.15,0.85], plot_datapoints=False)
plt.savefig("figs/" + plot_str + "_physical_corner_noweight.png", bbox_inches="tight")
fig = corner.corner(df[labels], labels=labels2, quantiles=[0.15,0.85], plot_datapoints=False, weights=df["probstability"])
plt.savefig("figs/" + plot_str + "_physical_corner_weight.png", bbox_inches="tight")

labels = ["Z12", "Z23", "m1+m2", "m2+m3", "total m"]
labels2 = [r"$Z_{12}$", r"$Z_{23}$", r"$m_1 + m_2$", r"$m_2 + m_3$", r"$m_1 + m_2 + m_3$"]
fig = corner.corner(df[labels], labels=labels2, quantiles=[0.15,0.85], plot_datapoints=False)
plt.savefig("figs/" + plot_str + "_param_corner_noweight.png", bbox_inches="tight")
fig = corner.corner(df[labels], labels=labels2, quantiles=[0.15,0.85], plot_datapoints=False, weights=df["probstability"])
plt.savefig("figs/" + plot_str + "_param_corner_weight.png", bbox_inches="tight")

sf.create_stab_hist(plot_str, df, "m1", label2=r"$m_1$", xlabel=r"$M_{\oplus}$")
sf.create_stab_hist(plot_str, df, "m2", label2=r"$m_2$", xlabel=r"$M_{\oplus}$")
sf.create_stab_hist(plot_str, df, "m3", label2=r"$m_3$", xlabel=r"$M_{\oplus}$")

sf.create_stab_hist(plot_str, df, "m1+m2", label2=r"$m_1 + m_2$", xlabel=r"$M_{\oplus}$")
sf.create_stab_hist(plot_str, df, "m2+m3", label2=r"$m_2 + m_3$", xlabel=r"$M_{\oplus}$")
sf.create_stab_hist(plot_str, df, "total m", label2=r"$m_1 + m_2 + m_3$", xlabel=r"$M_{\oplus}$")

sf.create_stab_hist(plot_str, df, "e1", show_quantiles=False, label2=r"$e_1$")
sf.create_stab_hist(plot_str, df, "e2", show_quantiles=False, label2=r"$e_2$")
sf.create_stab_hist(plot_str, df, "e3", show_quantiles=False, label2=r"$e_3$")

sf.create_stab_hist(plot_str, df, "Z12", show_quantiles=False, label2=r"$Z_{12}$")
sf.create_stab_hist(plot_str, df, "Z23", show_quantiles=False, label2=r"$Z_{23}$")
