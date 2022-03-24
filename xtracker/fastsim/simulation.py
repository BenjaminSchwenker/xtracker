# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.



import numpy as np
import random
import math
import scipy.optimize as opt


class MagneticField(object):
    """ A constant Magnetic field along z axis. Unit is Tesla.
    """

    def __init__(self, Bz=1.5):
        self.Bz = Bz


class Detector(object):
    """ A 2D tracking volume around the orign filled with sensors of finite resolution.
        The Layers are defined by a tuple containing radius and sigma in mm and time in ns
    """

    def __init__(self, layers=[], wlayers=[], width=2200, height=2200, length=2200, time=100):
        self.width = width
        self.height = height
        self.length = length
        self.time = time

    def setVXDLayers(self):
        self.layers = []
        self.layers.append((14, 0.001))
        self.layers.append((22, 0.001))
        self.layers.append((38, 0.001))
        self.layers.append((80, 0.001))
        self.layers.append((100, 0.001))
        self.layers.append((135, 0.001))

        self.vxdmin = 14
        self.vxdmax = 135

    def setCDCLayers(self):
        slayers = []
        slayers.append({'nlayer': 8, 'ncells': 160, 'rmin': 168, 'rmax': 238})
        slayers.append({'nlayer': 6, 'ncells': 160, 'rmin': 257, 'rmax': 348})
        slayers.append({'nlayer': 6, 'ncells': 192, 'rmin': 365, 'rmax': 455})
        slayers.append({'nlayer': 6, 'ncells': 224, 'rmin': 477, 'rmax': 567})
        slayers.append({'nlayer': 6, 'ncells': 256, 'rmin': 584, 'rmax': 674})
        slayers.append({'nlayer': 6, 'ncells': 288, 'rmin': 695, 'rmax': 785})
        slayers.append({'nlayer': 6, 'ncells': 320, 'rmin': 802, 'rmax': 892})
        slayers.append({'nlayer': 6, 'ncells': 352, 'rmin': 913, 'rmax': 1003})
        slayers.append({'nlayer': 6, 'ncells': 384, 'rmin': 1020, 'rmax': 1111})

        self.cdcmin = 168
        self.cdcmax = 1111

        self.wlayers = []
        for slayer in slayers:
            rmin = slayer['rmin']
            rmax = slayer['rmax']
            nlayer = slayer['nlayer']
            ncells = slayer['ncells']

            dr = (rmax - rmin) / float(nlayer - 1)
            for i in range(nlayer):
                r = rmin + i * dr
                off = i % 2 * math.pi / float(ncells)
                self.wlayers.append((r, ncells, off))

    def getNVXDLayers(self):
        return len(self.layers)

    def getNCDCLayers(self):
        return len(self.wlayers)


class Particle(object):
    """ Represents a charged particle in the detector.
    """

    def __init__(self, p, phi, theta, x0=0, y0=0, z0=0, charge=1, mass=0.0005):
        self.p = p
        self.phi = phi
        self.theta = theta
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.charge = charge
        self.mass = mass
        self.px = p * np.sin(theta) * np.cos(phi)
        self.py = p * np.sin(theta) * np.sin(phi)
        self.pz = p * np.cos(theta)
        self.pt = np.sqrt(self.px**2 + self.py**2)
        self.E = np.sqrt(mass**2 + p**2)

    def toArray(self):
        return [self.p, self.phi, self.theta, self.x0, self.y0, self.z0, self.charge, self.mass]

    def getDelta(self, scale=1):
        return scale * math.cos(self.phi), scale * math.sin(self.phi)

    def __str__(self):
        ret_value = """
            Particle(p={self.p}, phi={self.phi}, theta={self.theta}, x0={self.x0}, \
                y0={self.y0}, z0={self.z0}, charge={self.charge}, mass={self.mass})
            """.format(self=self)
        return ret_value


class ParticleGun(object):
    """ Samples the transverse momentum and polar angle of charged particles
        in an event.
    """

    def __init__(
        self, pMin=0.03, pMax=4.0, charge=None, mass=0.0005,
        phiMin=0, phiMax=2 * math.pi, thetaMin=0.2967, thetaMax=2.62
    ):
        self.pMin = pMin
        self.pMax = pMax
        self.phiMin = phiMin
        self.phiMax = phiMax
        self.thetaMin = thetaMin
        self.thetaMax = thetaMax
        self.ipX = 0
        self.ipY = 0
        self.ipZ = 0
        self.ipSigmaX = 0
        self.ipSigmaY = 0
        self.ipSigmaZ = 0
        self.charge = charge
        self.mass = mass

    def shoot(self):
        if self.charge is None:
            charge = random.choice([-1, +1])
        else:
            charge = self.charge
        return Particle(
            p=random.uniform(
                self.pMin, self.pMax), phi=random.uniform(
                self.phiMin, self.phiMax), theta=random.uniform(
                self.thetaMin, self.thetaMax), x0=0, y0=0, z0=0, charge=charge)


class Digitization(object):
    """ Creates a helical track and hits from particles
    """

    def __init__(self, magnetic_field, detector):
        self.magnetic_field = magnetic_field
        self.detector = detector

    def get_circle(self, particle):
        circR = particle.pt / self.magnetic_field.Bz / abs(particle.charge) / 0.3 * 1000
        circX = particle.x0 + np.sign(particle.charge) * circR * math.sin(particle.phi)
        circY = particle.y0 - np.sign(particle.charge) * circR * math.cos(particle.phi)
        return (circX, circY, abs(circR))

    def get_period(self, particle):
        circR = particle.pt / self.magnetic_field.Bz / abs(particle.charge) / 0.3 * 1000
        return 2 * np.pi * circR * particle.E / particle.pt / 300

    def get_noise(self, layers=range(6)):

        layer_id = random.choice(layers)

        r = self.detector.layers[layer_id][0]
        phi = random.uniform(0, 2 * np.pi)
        z = random.uniform(-self.detector.length * 0.5, self.detector.length * 0.5)
        t = random.uniform(-self.detector.time * 0.5, self.detector.time * 0.5)
        return (r * math.cos(phi), r * math.sin(phi), z, t)

    def get_hits(self, particle, rmax=1111):
        circX, circY, circR = self.get_circle(particle)
        x0, y0, z0 = particle.x0, particle.y0, particle.z0
        T = self.get_period(particle)

        def timeOfFlight(t): return T * t / np.pi / 2
        def trajx(t): return circR * np.cos(-1 * particle.charge * (t)) + circX
        def trajy(t): return circR * np.sin(-1 * particle.charge * (t)) + circY
        def trajz(t): return t * particle.pz / particle.mass
        def radius(t): return np.sqrt(trajx(t)**2 + trajy(t)**2)

        t0_x1 = opt.brentq(lambda t: (trajx(t) - x0), 0, np.pi)
        t0_x2 = opt.brentq(lambda t: (trajx(t) - x0), np.pi, 2 * np.pi)

        if abs(trajy(t0_x1) - y0) < abs(trajy(t0_x2) - y0):
            t0 = t0_x1
        else:
            t0 = t0_x2

        dr_max = 1.0
        N = circR * 2 * np.pi / dr_max
        dt = 2 * np.pi / N

        hit_look = False
        vxd_hits_bounds = []
        cdc_hits_bounds = []

        for i in range(int(N)):

            # check hit look
            if hit_look:
                hit_look = False
                continue

            # step along the trjectory
            t = i * dt
            rt = radius(t + t0)

            # check if particle left tracking region
            if rt > rmax:
                break

            # check hits in vxd
            for rl, sigmal in self.detector.layers:
                if abs(rt - rl) < dt * circR:
                    if rt - rl < 0:
                        if radius(t + t0 + dt) - rl > 0:
                            vxd_hits_bounds.append((rl, t, t + dt))
                            hit_look = True
                    elif rt - rl > 0:
                        if radius(t + t0 + dt) - rl < 0:
                            vxd_hits_bounds.append((rl, t, t + dt))
                            hit_look = True

            for rl, ncells, off in self.detector.wlayers:
                if abs(rt - rl) < dt * circR:
                    if rt - rl < 0:
                        if radius(t + t0 + dt) - rl > 0:
                            cdc_hits_bounds.append((rl, t, t + dt))
                            hit_look = True
                    elif rt - rl > 0:
                        if radius(t + t0 + dt) - rl < 0:
                            cdc_hits_bounds.append((rl, t, t + dt))
                            hit_look = True

        vxd_hits = []
        for rl, tmin, tmax in vxd_hits_bounds:
            troot = opt.brentq(lambda t: radius(t + t0) - rl, tmin, tmax)
            vxd_hits.append((trajx(troot + t0), trajy(troot + t0), trajz(timeOfFlight(troot)), timeOfFlight(troot)))

        cdc_hits = []
        for rl, tmin, tmax in cdc_hits_bounds:
            troot = opt.brentq(lambda t: radius(t + t0) - rl, tmin, tmax)
            cdc_hits.append((trajx(troot + t0), trajy(troot + t0), trajz(timeOfFlight(troot)), timeOfFlight(troot)))

        return vxd_hits, cdc_hits

    def getVXDLayerID(self, hit):
        x, y = hit[0], hit[1]
        return np.searchsorted([layer[0] for layer in self.detector.layers], np.sqrt(x**2 + y**2) - 1)

    def getCDCLayerID(self, hit):
        x, y = hit[0], hit[1]
        return np.searchsorted([layer[0] for layer in self.detector.wlayers], np.sqrt(x**2 + y**2) - 1)
