#!/usr/bin/env python
# coding: utf-8

# In[1]:


import stability_functions as sf
import numpy as np
from numpy.random import normal, seed, uniform
import os
import time
import sys
import rebound
import mr_forecast as mr
import numpy.random as rd
import radvel
import corner
import copy
import scipy
import pandas as pd
from scipy import optimize
from radvel.plot import orbit_plots
# plotting
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


seconds_p_day = 86400
days_p_year = 365.25
meters_p_AU = 149597870700
earth_mass_2_solar_mass = 0.000003003
year_p_reboundtime = 1 / (2 * np.pi)


# In[3]:


if sys.argv[1] != "-f":
    err_ind = int(sys.argv[1])
else:
    err_ind = -1
df = pd.read_pickle("mcmc/mcmc_%d.pkl"%err_ind)
df = df.drop(columns="jit")
df = df.drop(columns="lnprobability")


# In[4]:


inds = np.sort(np.random.choice(range(len(df)), size=int(np.floor(len(df)/1000)), replace=False))


# In[6]:


for j in range(len(inds)):
    
#     if sys.argv[1] != "-f":
#         params_df = df.iloc[int(sys.argv[2])]
#     else:
#         params_df = df.iloc[0]
    
    params_df = df.iloc[inds[j]]
    
    Nplanets = 3
    params = radvel.Parameters(Nplanets, basis='per tc secosw sesinw k')
    for col in df.columns:
        params[col] = radvel.Parameter(value=params_df[col])
    new_params = params.basis.to_any_basis(params, 'per tp e w k')

    #set up simulation
    sim = rebound.Simulation()

    sim.collision = 'line'

    sim.collision_resolve = sf.collision

    #add star
    Mstar = 1.145
    sim.add(m=Mstar)

    incs = np.pi / 180 * np.array([sf.a_normal(85.5, 1.5, 0.5), sf.a_normal(86.23, 0.26, 0.26), sf.a_normal(87.43, 0.18, 0.19)])

    for i in range(3):
        P = float(new_params["per%d"%(i+1)]) / days_p_year
        e = float(new_params["e%d"%(i+1)]) # Omega = W  # longitude of ascending node 
        T = float(new_params["tp%d"%(i+1)])
        pomega = float(new_params["w%d"%(i+1)])  # longitude of periastron  
        inc = incs[i]
        m = sf.mass_from_VSA(P, Mstar, float(new_params["k%d"%(i+1)]), e, inc) * earth_mass_2_solar_mass

        # omega = w
        # M = M
        r = (1 - e) * np.cbrt(P * P * m * earth_mass_2_solar_mass / Mstar / 3) / 2

        # sim.add(m=m, P=P / year_p_reboundtime, e=e, inc=inc - np.pi/2, Omega=W, omega=w, M=M, r=radii[i]) #G=1 units!
        sim.add(m=m, P=P / year_p_reboundtime, e=e, inc=inc - np.pi/2, T=T, pomega=pomega, r=r) #G=1 units!

    sim.move_to_com()
    
    sf.stability_score(sim)  # dummy score
    
    # initialize list of lists 
    data = [[inds[j], sf.stability_score(sim)]] 

    if j==0:
        save_df = pd.DataFrame(data, columns = ['ind', 'score'])
    else:
        save_df = save_df.append(pd.DataFrame(data, columns = ['ind', 'score']))
        


# In[7]:


save_df.to_pickle("stab_scores_%d.pkl"%err_ind)
# pd.read_pickle("stab_scores_%d.pkl"%err_ind)

