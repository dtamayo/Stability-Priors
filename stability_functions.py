import numpy as np
import numpy.random as rd
import scipy
import os
import os.path
import rebound
import mr_forecast as mr
# Dan's stability model packages
import random
import dill
import sys
import time
import pandas as pd
if not '../MLstability/generate_training_data' in sys.path:
    sys.path.append('../MLstability/generate_training_data')
# from training_data_functions import ressummaryfeaturesxgb
from training_data_functions import ressummaryfeaturesxgbv6

folderpath = '../MLstability'
# model = 'ressummaryfeaturesxgb_resonantAMD.pkl'
model = 'ressummaryfeaturesxgbv6_resonant.pkl'
model, features, featurefolder = dill.load(open(folderpath+'/models/'+model, "rb"))

seconds_p_day = 86400
days_p_year = 365.25
meters_p_AU = 149597870700
earth_mass_2_solar_mass = 0.000003003
year_p_reboundtime = 1 / (2 * np.pi)

def build_sim(Ps, ms, es=0, incs=0, Mstar=1):
    
    Nplanets = len(Ps)
    
    if all(es==0):
        es = np.zeros(Nplanets)
        
    assert Nplanets==len(ms)==len(es)

    radii = np.zeros(Nplanets)
    for i in range(Nplanets):
        radii[i] = (1 - es[i]) * np.cbrt(Ps[i] * Ps[i] * ms[i] * earth_mass_2_solar_mass / Mstar / 3) / 2

    #set up simulation
    sim = rebound.Simulation()

    sim.collision = 'line'

    #add star
    sim.add(m=Mstar)

    if all(incs==0):
        a = np.cbrt((Ps[i]*(2*np.pi))**2 * Ms)
        Rs2AU = 0.00465047
        Rstar = 1
        incs=np.array([rd.uniform(high=0.9 * Rstar * Rs2AU / a[i]) for i in range(Nplanets)])

    Ws=2 * np.pi * rd.sample(Nplanets)
    ws=2 * np.pi * rd.sample(Nplanets)
    Ms=2 * np.pi * rd.sample(Nplanets)
    for i in range(Nplanets):
        m = ms[i] * earth_mass_2_solar_mass
        P = Ps[i]
        e = es[i]
        inc = incs[i]
        W = Ws[i]
        w = ws[i]
        M = Ms[i]
        sim.add(m=m, P=P / year_p_reboundtime, e=e, inc=inc, Omega=W, omega=w, M=M, r=radii[i]) #G=1 units!
    sim.move_to_com()
    
    sim.collision_resolve = collision

    return sim

def rebound_rvs(times, sim):
    
    assert np.all(np.diff(times)>=0), "times passed aren't sorted"
    
    times_rb = times / year_p_reboundtime
    
    rvs = np.zeros(len(times))
    for i in range(len(times)):
        sim.integrate(times_rb[i])
        rvs[i] = sim.particles[0].vx
    rvs = rvs * meters_p_AU / year_p_reboundtime / days_p_year / seconds_p_day
    return rvs
    
def a_normal(mean, upper_sigma, lower_sigma):
    draw = rd.normal()
    if draw > 0:
        draw *= upper_sigma
    else:
        draw *= lower_sigma
    return draw + mean

def draw_masses(sample=int(1e3 * 10)):
    mb = scipy.stats.gaussian_kde(mr.Rstat2M(2.085, 0.066, return_post=True, sample_size=sample, grid_size=sample))
    mc = scipy.stats.gaussian_kde(mr.Rstat2M(1.939, 0.069, return_post=True, sample_size=sample, grid_size=sample))
    md = scipy.stats.gaussian_kde(mr.Rstat2M(2.164, 0.085, return_post=True, sample_size=sample, grid_size=sample))
    return mb, mc, md

def build_chosen_HR858(ms, es, return_extra=False, Ps = np.array([a_normal(3.58599, 0.00015, 0.00015), a_normal(5.97293, 0.00060, 0.00053), a_normal(11.2300, 0.0011, 0.0010)]) / days_p_year, incs = np.pi / 180 * np.array([a_normal(85.5, 1.5, 0.5), a_normal(86.23, 0.26, 0.26), a_normal(87.43, 0.18, 0.19)])):
    sim = build_sim(Ps, ms, Mstar=1.145, es=es, incs=incs - np.pi/2)
    if return_extra:
        return sim, Ps, ms, es, incs
    else:
        return sim

def build_HR858(mb, mc, md, es=np.array([rd.rand() * 0.3, rd.rand() * 0.19, rd.rand() * 0.28]), return_extra=False):
    ms = np.array([mb.resample(1)[0][0], mc.resample(1)[0][0], md.resample(1)[0][0]])  # earth masses
    return build_chosen_HR858(ms, es, return_extra=return_extra)
    
def stability_score(sim):
    args = (10000, 1000) # (Norbits, Nout) Keep this fixed
    summaryfeatures = ressummaryfeaturesxgbv6(sim, args)
    if features is not None:
        summaryfeatures = summaryfeatures[features]
    summaryfeatures = pd.DataFrame([summaryfeatures])
    return model.predict_proba(summaryfeatures)[:, 1][0]

#if a collision occurs, end the simulation
def collision(reb_sim, col):
    reb_sim.contents._status = 5 # causes simulation to stop running and have flag for whether sim stopped due to collision
    return 0

def replace_snapshot(sim, filename):
    if os.path.isfile(filename):
        os.remove(filename)
    sim.simulationarchive_snapshot(filename)
    
def VSA(P, m_star, m_planet, e, i):
    m_planet *= earth_mass_2_solar_mass
    comb_mass = m_star + m_planet
    return 2 * np.pi * np.sin(i) * m_planet / (np.sqrt(1 - (e * e)) * np.cbrt(comb_mass * comb_mass * P))* meters_p_AU / seconds_p_day / days_p_year

def mass_from_VSA(P, m_star, VSA, e, i):
    return 1 / (2 * np.pi * np.sin(i) * VSA / (np.sqrt(1 - (e * e)) * np.cbrt(m_star * m_star * P))* meters_p_AU / seconds_p_day / days_p_year) / earth_mass_2_solar_mass