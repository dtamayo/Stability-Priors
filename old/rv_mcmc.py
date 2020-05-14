import stability_functions as sf
import numpy as np
import rebound
import numpy.random as rd
import radvel
import corner
import copy
import scipy
import pandas as pd
from scipy import optimize
from radvel.plot import orbit_plots
import sys
# # plotting
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
matplotlib.use('Agg')
# # %matplotlib inline

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
# system = systems[0]
system = systems[int(sys.argv[1])]
sf.valid_system(system)
sim_names = system + "/" + system
systems_ran = 256

df = pd.read_csv(sim_names + ".csv", index_col=0)
df = df.sort_values("probstability", ascending=False)
nsim_l = np.array(df.iloc[0:(systems_ran - 1)]["sim"]).astype(int)

df1 = pd.DataFrame(data=nsim_l, columns=["sim"])
df1 = df1.set_index("sim")
df1["stable"] = False
for nsim in nsim_l:
    try:
        df2 = pd.read_csv(sim_names + "_%d_orbits.csv"%nsim, index_col=0)
        df1.stable[nsim] = df2["stable"][0]
    except:
        df1.stable[nsim] = False

stable_sims = np.array(df1[df1.stable == True].index).astype(int)

e_df = df.loc[stable_sims]
e_df["total e"] = e_df["e1"] + e_df["e2"] + e_df["e3"]
e_df = e_df.sort_values("total e")
sim_nums = np.array(e_df["sim"]).astype(int)

def myPriorFunc(inp_list):
    return sf.myPriorFunc(labels, Mstar, prior_ms_small, prior_es_small, inp_list)

fig_name = "figs/" + system + "_"

sim_num = sim_nums[0]
name = sim_names+"_sa_"+str(sim_num)+".bin"

print("simulation archive:", name)
sim = rebound.SimulationArchive(name)[0]

n_meas = 50
t_rb = 0.75 * np.sort(rd.rand(n_meas))
syn_rv_base = sf.rebound_rvs(t_rb, rebound.SimulationArchive(name)[0])
ti_rb = np.linspace(np.min(t_rb), np.max(t_rb), 1000)
syn_rv_no_noise = sf.rebound_rvs(ti_rb, rebound.SimulationArchive(name)[0])
t = t_rb * sf.days_p_year
ti = ti_rb * sf.days_p_year

errs = np.array([30, 10, 3, 1, 0.3, 0.1])
jitter = 0.0
stellar = jitter * rd.randn(n_meas)

print("jitter value:", jitter)

Ps = np.array([sim.particles[i+1].P for i in range(3)]) * (sf.days_p_year * sf.year_p_reboundtime)
es = np.array([sim.particles[i+1].e for i in range(3)])
Mstar = sim.particles[0].m
ms = np.array([sim.particles[i+1].m for i in range(3)]) * sf.earth_mass_p_solar_mass
incs = np.array([sim.particles[i+1].inc for i in range(3)]) + np.pi/2
og_Ks = sf.VSA(Ps / sf.days_p_year, Mstar, ms, es, incs)
Ks = np.copy(og_Ks)

omegas = np.array([sim.particles[i+1].omega for i in range(3)])
print("og Ks:", og_Ks)
print("initial Ks:", Ks)

truths = list(es)
truths.extend([sim.particles[i+1].m * sf.earth_mass_p_solar_mass for i in range(3)])
    
labels_base = ["e1","e2", "e3", "m1", "m2", "m3"]
labels2 = [r"$e_1$", r"$e_2$", r"$e_3$", r"$m_1$", r"$m_2$", r"$m_3$"]

# err_ind = -1
err_ind = int(sys.argv[2])

errval = errs[err_ind]

print("working on err_ind %d\n"%err_ind)
print("with an error value of %f m/s\n"%errval)

syn_rv = syn_rv_base + rd.randn(n_meas) * errval + stellar

params = radvel.Parameters(3, basis='per tc secosw sesinw k')
params['per1'] = radvel.Parameter(value=Ps[0])
params['tc1'] = radvel.Parameter(value= 2458409.18969)
params['secosw1'] = radvel.Parameter(value=np.sqrt(es[0]) * np.cos(omegas[0]))
params['sesinw1'] = radvel.Parameter(value=np.sqrt(es[0]) * np.sin(omegas[0]))
params['k1'] = radvel.Parameter(value=Ks[0])

params['per2'] = radvel.Parameter(value=Ps[1])
params['tc2'] = radvel.Parameter(value=2458415.6344)
params['secosw2'] = radvel.Parameter(value=np.sqrt(es[1]) * np.cos(omegas[1]))
params['sesinw2'] = radvel.Parameter(value=np.sqrt(es[1]) * np.sin(omegas[1]))
params['k2'] = radvel.Parameter(value=Ks[1])

params['per3'] = radvel.Parameter(value=Ps[2])
params['tc3'] = radvel.Parameter(value=2458409.7328)
params['secosw3'] = radvel.Parameter(value=np.sqrt(es[2]) * np.cos(omegas[2]))
params['sesinw3'] = radvel.Parameter(value=np.sqrt(es[2]) * np.sin(omegas[2]))
params['k3'] = radvel.Parameter(value=Ks[2])

# params['dvdt'] = radvel.Parameter(value=0)
# params['curv'] = radvel.Parameter(value=0)

rv_mod = radvel.RVModel(params)

like_syn = radvel.likelihood.RVLikelihood(rv_mod, t, syn_rv, np.zeros(t.size)+errval)
like_syn.params['gamma'] = radvel.Parameter(value=0)
like_syn.params['jit'] = radvel.Parameter(value=jitter)

like_syn.params['jit'].vary = False # Don't vary jitter
# like_syn.params['k1'].vary = False # Don't vary period
# like_syn.params['k2'].vary = False # Don't vary period
# like_syn.params['k3'].vary = False # Don't vary period
# like_syn.params['per1'].vary = False # Don't vary period
# like_syn.params['per2'].vary = False # Don't vary period
# like_syn.params['per3'].vary = False # Don't vary period
like_syn.params['dvdt'].vary = False # Don't vary dvdt
like_syn.params['curv'].vary = False # Don't vary curvature
like_syn.params['gamma'].vary = False # Don't vary gamma
print(like_syn, "\n")

# Plot initial model
sf.plot_radvel_results(like_syn, t)
plt.plot(ti, syn_rv_no_noise)
plt.savefig(fig_name + "%d_beforeMLE.png"%err_ind, bbox_inches="tight")
plt.close(fig="all")

res  = optimize.minimize(like_syn.neglogprob_array, like_syn.get_vary_params(), method='L-BFGS-B')
# print(res)
print(like_syn, "\n")
sf.plot_radvel_results(like_syn, t) # plot best fit model
plt.plot(ti, syn_rv_no_noise)
plt.savefig(fig_name + "%d_afterMLE.png"%err_ind, bbox_inches="tight")

post = radvel.posterior.Posterior(like_syn)

if system == "HR858":
    post.priors += [radvel.prior.Gaussian("per1", 3.58599, 0.00015)]
    post.priors += [radvel.prior.Gaussian("per2", 5.97293, 0.00057)]
    post.priors += [radvel.prior.Gaussian("per3", 11.2300, 0.0011)]

if system == "K431":
    post.priors += [radvel.prior.Gaussian("per1", 6.80252171, 7.931e-05)]
    post.priors += [radvel.prior.Gaussian("per2", 8.70337044, 9.645e-05)]
    post.priors += [radvel.prior.Gaussian("per3", 11.9216214, 0.0001182)]

if system == "TOI270":
    post.priors += [radvel.prior.Gaussian("per1", 3.36008, 0.000068)]
    post.priors += [radvel.prior.Gaussian("per2", 5.660172, 0.000035)]
    post.priors += [radvel.prior.Gaussian("per3", 11.38014, 0.00011)]

if system == "L98-59":
    post.priors += [radvel.prior.Gaussian("per1", 2.25314, 0.00002)]
    post.priors += [radvel.prior.Gaussian("per2", 3.690621, 0.0000135)]
    post.priors += [radvel.prior.Gaussian("per3", 7.45086, 0.000045)]

#Holczer et al. 2016
if system == "K23":
    post.priors += [radvel.prior.Gaussian("per1", 7.10697755, 0.00001048)]
    post.priors += [radvel.prior.Gaussian("per2", 10.74244253, 0.00000349)]
    post.priors += [radvel.prior.Gaussian("per3", 15.27458613, 0.00000510)]

labels = [k for k in post.params.keys() if post.params[k].vary]

prior_ms = [scipy.stats.gaussian_kde(df["m1"]), scipy.stats.gaussian_kde(df["m2"]), scipy.stats.gaussian_kde(df["m3"])]
prior_es = [scipy.stats.gaussian_kde(df["e1"]), scipy.stats.gaussian_kde(df["e2"]), scipy.stats.gaussian_kde(df["e3"])]

inds = np.sort(np.random.choice(range(len(df)), size=4000, replace=False))
prior_ms_small = [scipy.stats.gaussian_kde(df["m1"][inds]), scipy.stats.gaussian_kde(df["m2"][inds]), scipy.stats.gaussian_kde(df["m3"][inds])]
prior_es_small = [scipy.stats.gaussian_kde(df["e1"][inds]), scipy.stats.gaussian_kde(df["e2"][inds]), scipy.stats.gaussian_kde(df["e3"][inds])]

post.priors += [radvel.prior.UserDefinedPrior(labels, myPriorFunc, 'Eccentricty and mass prior')]
# post.priors += [radvel.prior.EccentricityPrior(3)]

print("Before fitting\n")
print(post, "\n")

# res  = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Nelder-Mead' )
res  = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='L-BFGS-B')
# sf.plot_radvel_results(post.likelihood, t)
plt.savefig(fig_name + "%d_afterpriors.png"%err_ind, bbox_inches="tight")
print("After fitting\n")
print(post, "\n")

RVPlot = orbit_plots.MultipanelPlot(post)
RVPlot.plot_multipanel()
# plt.show()
plt.savefig(fig_name + "%d_afterpriors_multi.png"%err_ind, bbox_inches="tight")
# plt.close(fig="all")

# df3 = radvel.mcmc(post) # amount of steps = nrun * 8 * nwalkers ?, default nrun is 10000, default nwalkers is 50?
df3 = radvel.mcmc(post, nrun=1000)
df3.to_pickle(fig_name + "%d_mcmc.pkl"%err_ind)

fig = corner.corner(df3[labels], labels=labels, quantiles=[0.16,0.84], plot_datapoints=False)
plt.savefig(fig_name + "%d_corner.png"%err_ind, bbox_inches="tight")

df2 = pd.read_pickle(fig_name + "%d_mcmc.pkl"%err_ind)

# df2 = df3.copy()
for i in range(1,4):
    df2["e%d"%i] = df2["secosw%d"%i] * df2["secosw%d"%i] + df2["sesinw%d"%i] * df2["sesinw%d"%i]
    df2["m%d"%i] = sf.mass_from_VSA(df2["per%d"%i] / sf.days_p_year, Mstar, df2["k%d"%i].abs(), df2["e%d"%i], np.pi/2)
df2 = df2[labels_base]

fig = corner.corner(df2, labels=labels2, truths=truths, quantiles=[0.16,0.84], plot_datapoints=False)
plt.savefig(fig_name + "%d_physical_corner.png"%err_ind, bbox_inches="tight")

df2["m1+m2"] = df2["m1"] + df2["m2"]
df2["m2+m3"] = df2["m2"] + df2["m3"]
df2["total m"] = df2["m1"] + df2["m2"] + df2["m3"]

sf.create_mcmc_hist(system, err_ind, df, df2, "m1", label2=r"$m_1$", xlabel=r"$M_{\oplus}$")
sf.create_mcmc_hist(system, err_ind, df, df2, "m2", label2=r"$m_2$", xlabel=r"$M_{\oplus}$")
sf.create_mcmc_hist(system, err_ind, df, df2, "m3", label2=r"$m_3$", xlabel=r"$M_{\oplus}$")

sf.create_mcmc_hist(system, err_ind, df, df2, "m1+m2", label2=r"$m_1 + m_2$", xlabel=r"$M_{\oplus}$")
sf.create_mcmc_hist(system, err_ind, df, df2, "m2+m3", label2=r"$m_2 + m_3$", xlabel=r"$M_{\oplus}$")
sf.create_mcmc_hist(system, err_ind, df, df2, "total m", label2=r"$m_1 + m_2 + m_3$", xlabel=r"$M_{\oplus}$")

#     xs = np.linspace(0,1,1000)
sf.create_mcmc_hist(system, err_ind, df, df2, "e1", show_quantiles=False, label2=r"$e_1$")
sf.create_mcmc_hist(system, err_ind, df, df2, "e2", show_quantiles=False, label2=r"$e_2$")
sf.create_mcmc_hist(system, err_ind, df, df2, "e3", show_quantiles=False, label2=r"$e_3$")
    
#     sys.stdout.close()

# sys.stdout = std_out_holder


# In[ ]:




