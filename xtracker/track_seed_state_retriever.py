##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import numpy as np
from math import sqrt
from ROOT import TMatrixDSym, TVectorD


def findCircleRadius(x1, y1, x2, y2, x3, y3):
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius
    r = sqrt(sqr_of_r)

    return r


def calcPt(radius):
    Bz = 1.5
    return Bz * radius * 0.00299792458


def calcCurvatureSignum(x):
    """Calculate curvature based on triplets of measurements.
        Ignores uncertainties.
        Returns -1,0,1 depending on the sum of all triplets.
    """

    if x.shape[0] < 3:
        return 0

    ab = x[1, :] - x[0, :]
    bc = x[2, :] - x[1, :]
    sumOfCurvature = bc[1] * ab[0] - bc[0] * ab[1]

    if 0 <= sumOfCurvature:
        return 1
    else:
        return -1


def getSeedState(x):

    # Charge of track from sign of circle defined by hits x
    charge = -1 * calcCurvatureSignum(x)

    # Radius of circle defined by hits x
    rho = findCircleRadius(x1=x[0, 0], y1=x[0, 1], x2=x[1, 0], y2=x[1, 1], x3=x[2, 0], y3=x[2, 1])

    # Compute momenta
    pT = calcPt(rho)
    momVec = x[1, :] - x[0, :]
    momVec[2] = 0
    dr = np.linalg.norm(momVec)
    momVec = pT * momVec / dr
    tanLambda = (x[1, 2] - x[0, 2]) / dr
    pZ = pT * tanLambda

    # Covariance matrix of seed state
    covSeed = TMatrixDSym(6)
    covSeed.Zero()  # just to be save
    covSeed[0][0] = 1
    covSeed[1][1] = 1
    covSeed[2][2] = 2 * 2
    covSeed[3][3] = 0.1 * 0.1
    covSeed[4][4] = 0.1 * 0.1
    covSeed[5][5] = 0.2 * 0.2

    # Seed state
    stateSeed = TVectorD(6)

    # XYZ position of first hit along track
    stateSeed[0] = x[0, 0]
    stateSeed[1] = x[0, 1]
    stateSeed[2] = x[0, 2]

    if charge == 0:
        stateSeed[3] = 0
        stateSeed[4] = 0
        stateSeed[5] = 0
    else:
        stateSeed[3] = momVec[0]
        stateSeed[4] = momVec[1]
        stateSeed[5] = pZ

    return stateSeed, covSeed, charge


def computeHelixNorm(x):
    """Compute how close N 3D hits forming a Nx3 array x resample a helix.

    Based on https://www.kaggle.com/danieleewww/nievergelt-helix-fitting
    """

    # compute average
    xm = np.mean(x)

    # form matrix
    X = (x - xm).T

    # compute singular values and eigenvectors
    v, s, t = np.linalg.svd(X, full_matrices=True)

    sigma1 = s[0]
    sigma2 = s[1]
    sigma3 = s[2]
    v1 = t[0]
    v2 = t[1]
    v3 = t[2]

    # fitting the axis and radius of the helix
    Z = np.zeros((x.shape[0], 10), np.float32)
    Z[:, 0] = x[:, 0]**2
    Z[:, 1] = 2 * x[:, 0] * x[:, 1]
    Z[:, 2] = 2 * x[:, 0] * x[:, 2]
    Z[:, 3] = 2 * x[:, 0]
    Z[:, 4] = x[:, 1]**2
    Z[:, 5] = 2 * x[:, 1] * x[:, 2]
    Z[:, 6] = 2 * x[:, 1]
    Z[:, 7] = x[:, 2]**2
    Z[:, 8] = 2 * x[:, 2]
    Z[:, 9] = 1

    v, s, t = np.linalg.svd(Z, full_matrices=True)
    smallest_value = np.min(np.array(s))
    smallest_index = np.argmin(np.array(s))
    T = np.array(t)
    T = T[smallest_index, :]
    S = np.zeros((4, 4), np.float32)
    S[0, 0] = T[0]
    S[0, 1] = S[1, 0] = T[1]
    S[0, 2] = S[2, 0] = T[2]
    S[0, 3] = S[3, 0] = T[3]
    S[1, 1] = T[4]
    S[1, 2] = S[2, 1] = T[5]
    S[1, 3] = S[3, 1] = T[6]
    S[2, 2] = T[7]
    S[2, 3] = S[3, 2] = T[8]
    S[3, 3] = T[9]
    norm = np.linalg.norm(np.dot(Z, T), ord=2)**2
