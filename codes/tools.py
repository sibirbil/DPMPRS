# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:29:12 2021

@author: Utku Karaca
"""
import numpy as np
from numpy import linalg as LA


class instance:
    # Coefficients for revenue and capacity vectors
    revenue_coef = 100  # each revenue vector (r_k) will be multiplied with this
    capacity_coef = 10  # each capacity vector (both c, c_k) will be multiplied with this

    def __init__(self, parties, m, max_private_cap, max_product, seed):
        self.parties = int(parties)
        self.m = int(m)
        self.seed = int(seed)
        self.max_private_cap = int(max_private_cap)
        self.max_product = int(max_product)
        np.random.seed(seed)
        # Parameters
        #     number of individual capacities
        self.m_parties = np.random.randint(low=5,
                                           high=max_private_cap + 1,
                                           size=parties)
        # number of products offered by parties
        self.n = np.random.randint(low=10,
                                   high=self.max_product + 1,
                                   size=self.parties)
    # Random Generation of
        #   A (made of A_k's),
        #   B (made of B_k's),
        #   r (made of r_k's),
        #   c_individual (made of c_k's),
        #   c
        self.r = 50 + instance.revenue_coef * np.random.rand(sum(self.n))
        self.A = 5 * np.random.rand(self.m, sum(self.n))
        self.B = np.zeros((sum(self.m_parties), sum(self.n)))
        for i in range(self.parties):
            self.party_matrix = np.zeros((self.m_parties[i], self.n[i]))
            if i == 0:
                self.party_matrix[0:self.m_parties[i], 0:self.n[i]] = np.random.rand(self.m_parties[i], self.n[i])
                self.B[0:self.m_parties[i], 0:self.n[i]] = self.party_matrix
            else:
                self.party_matrix[0:self.m_parties[i], 0:self.n[i]] = np.random.rand(self.m_parties[i], self.n[i])
                self.B[sum(self.m_parties[0:i]):(sum(self.m_parties[0:i]) + self.m_parties[i]),
                       sum(self.n[0:i]):(sum(self.n[0:i]) + self.n[i])] = self.party_matrix
        self.c = 10 + instance.capacity_coef * np.random.rand(self.m)
        self.c_individual = 10 + instance.capacity_coef * np.random.rand(sum(self.m_parties))
        # Compute norm of c and divide A by norm(c)
        self.A = self.A / (LA.norm(self.c))
        self.c = self.c / LA.norm(self.c)


def sigmaCalculator(sensitivity, T, m, rho):
    return np.sqrt((sensitivity**2 * T * m) / (2 * rho))


def rhoCalculator(epsilon, delta, xi=0):
    return ((epsilon - xi)**2 / (4 * np.log(1/delta)))


def noisy_sk_sinan_hoca(sk_t1, noisySk_t, sensitivity_delta, rho, T, m, c):
    noisy_vector = np.zeros(len(c))
    for i in range(len(c)):
        d_k = sk_t1[i] - noisySk_t[i]
        d_k_tilde = max(sensitivity_delta / 2, (d_k / c[i]) * (sensitivity_delta / 2))
        noisy_vector[i] = (sk_t1[i] + d_k_tilde +
                           np.random.normal(loc=0,
                                            scale=sigmaCalculator(sensitivity_delta, T, m, rho)))
    return noisy_vector


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = 'float64'  # arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def psi(eps):
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(eps**2))


def Psi(x, n):
    return 0.5 + (1/np.sqrt(2*np.pi))*(sum(((-1)**k * (x**(2*k + 1)))/((2**k)*np.math.factorial(k)*(2*k+1))
                                           for k in range(n)))


def truncatedMean(mean, sigma, a, b):
    alfa = (a - mean)/sigma
    beta = (b - mean)/sigma
    return mean + (psi(alfa)-psi(beta))/(Psi(beta, 10)-Psi(alfa, 10))


def truncatedVar(mean, sigma, a, b):
    pass
