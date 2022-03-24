# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pandas as pd


import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import os
import math

import xtracker.fastsim.simulation as sim

# initialize the tracking detectIor
detector = sim.Detector()
detector.setVXDLayers()
detector.setCDCLayers()

# initialize the magnetic field
bfield = sim.MagneticField()

# initialize the simulator
digi = sim.Digitization(bfield, detector)


def make_event(n_tracks, n_noise, smear=True, normed=True, rmax=detector.vxdmax, phiMin=0, phiMax=0.5,
               pMin=0.11, pMax=0.7, thetaMin=0.2967, thetaMax=2.62, charge=None):

    # initialize the particle gun
    gun = sim.ParticleGun(pMin=pMin, pMax=pMax, charge=charge, phiMin=phiMin, phiMax=phiMax, thetaMin=thetaMin, thetaMax=thetaMax)

    all_hits = {'particle_id': [], 'layer': [], 'x': [], 'y': [], 'z': [], 't': [], 'hit_id': []}
    all_truth = {'hit_id': [], 'particle_id': [], 'weight': []}
    all_particles = {'vx': [], 'vy': [], 'vz': [], 'px': [], 'py': [], 'pz': [], 'q': [], 'nhits': [], 'particle_id': []}

    hit_id = 0

    for i in range(n_tracks):
        particle = gun.shoot()
        mctrack = digi.get_circle(particle)
        a_vxdhits, a_cdchits = digi.get_hits(particle, rmax=rmax)

        # Create a common list of hits
        hits = a_vxdhits  # + a_cdchits
        hits = np.array(hits)

        all_particles['particle_id'].append(i)
        all_particles['px'].append(particle.px)
        all_particles['py'].append(particle.py)
        all_particles['pz'].append(particle.pz)
        all_particles['vx'].append(particle.x0)
        all_particles['vy'].append(particle.y0)
        all_particles['vz'].append(particle.z0)
        all_particles['q'].append(particle.charge)
        all_particles['nhits'].append(hits.shape[0])

        for j in range(hits.shape[0]):
            all_hits['x'].append(hits[j, 0])
            all_hits['y'].append(hits[j, 1])
            all_hits['z'].append(hits[j, 2])
            all_hits['t'].append(hits[j, 3])
            all_hits['layer'].append(digi.getVXDLayerID(hits[j, 0:2]))
            all_hits['hit_id'].append(hit_id)
            all_hits['particle_id'].append(i)

            all_truth['hit_id'].append(hit_id)
            all_truth['particle_id'].append(i)
            all_truth['weight'].append(0)

            hit_id += 1

    # We need to invent mother particle for noise hits
    nid = -1

    if n_noise > 0:
        # We need to invent mother particle
        all_particles['particle_id'].append(nid)
        all_particles['px'].append(0)
        all_particles['py'].append(0)
        all_particles['pz'].append(0)
        all_particles['vx'].append(0.0)
        all_particles['vy'].append(0.0)
        all_particles['vz'].append(0.0)
        all_particles['q'].append(0)
        all_particles['nhits'].append(0)

    for i in range(n_noise):
        noise = digi.get_noise()
        all_hits['x'].append(noise[0])
        all_hits['y'].append(noise[1])
        all_hits['z'].append(noise[2])
        all_hits['t'].append(noise[3])
        all_hits['layer'].append(digi.getVXDLayerID(noise[0:2]))
        all_hits['hit_id'].append(hit_id)
        all_hits['particle_id'].append(nid)

        all_truth['hit_id'].append(hit_id)
        all_truth['particle_id'].append(nid)
        all_truth['weight'].append(0)

        hit_id += 1

    return pd.DataFrame(all_hits), pd.DataFrame(all_truth), pd.DataFrame(all_particles)
