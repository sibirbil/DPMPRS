# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:54:20 2020

@author: Utku Karaca
"""

from gurobipy import GRB, Model, GurobiError
import numpy as np
# from tools import sigmaCalculator, rhoCalculator, noisy_sk_sinan_hoca

# Each party solves the following problem (optimize_by_party)
#
#   MAXIMIZE    r_k*x_k
#   subject to
#               A_k * x_k <= s_k
#               B_k * x_k <= c_k
#               x_k, s_k  >=   0
#
# Returns the vector s_k and noisy s_k


def optimize_by_party(n_ind, r_ind, A_ind, B_ind, c_ind, demand_ind, c, rho_0,
                      lamda, noisySk_t, sensitivity_delta, S_val):
    try:
        optMod = Model('Model_5')
        x = optMod.addVars(int(n_ind), lb=0, name='x')
        s = optMod.addVars(len(lamda), lb=0, name='s')
        obj_fn = (sum(r_ind[i]*x[i] for i in range(n_ind)) -
                  sum(lamda[i]*s[i] for i in range(len(lamda)))
                  )
        optMod.setObjective(obj_fn, GRB.MAXIMIZE)

        for i in range(len(lamda)):
            optMod.addLConstr(sum(A_ind[i, j]*x[j] for j in range(n_ind)),
                              GRB.LESS_EQUAL,
                              s[i],
                              name=('Shared Capacity Constraint for Source ' + str(i+1))
                              )

        for i in range(len(lamda)):
            optMod.addLConstr(s[i], GRB.LESS_EQUAL, c[i],
                              name=('Auxiliary Shared Capacity Constraint for Source ' + str(i+1)))

        # for i in range(len(lamda)):
        #     optMod.addLConstr(s[i], GRB.LESS_EQUAL, upper_limit * c[i],
        #                       name = ('Upper Bound on Shared Capacity Constraint for Source ' + str(i+1)))

        for i in range(len(c_ind)):
            optMod.addLConstr(sum(B_ind[i, j]*x[j] for j in range(n_ind)),
                              GRB.LESS_EQUAL,
                              c_ind[i],
                              name=('Individual Capacity Constraint for Source ' +
                                    str(i+1)))

        for i in range(len(demand_ind)):
            optMod.addLConstr(-1 * x[i], GRB.LESS_EQUAL, demand_ind[i],
                              name=('Demand for Product ' + str(i+1)))

        optMod.setParam('OutputFlag', False)  # Prevent output logs in the command line.
        optMod.optimize()  # solve the model
        optMod.write('dp_model.lp')
        s_values = np.array(list(s[i].x for i in range(len(s))))
        #
        #
        # DP NOISE ADDITION
        #
        #    Laplace Mechanism
        # noise = np.random.laplace(loc=0, scale=(max_iter*S_val)/epsilon)
        #
        #    zCDP Mechanism
        dp_std = np.sqrt(((S_val**2)/(2 * rho_0)))
        noise = np.random.normal(loc=0, scale=dp_std)
        # addition
        s_values_noisy = s_values + noise  # np.minimum(s_values, S_val) + noise
        # truncation
        # truncated_s_values_noisy = np.minimum(np.maximum(np.ones(len(c))*0.001, s_values_noisy), c)
        truncated_s_values_noisy = s_values_noisy
        return optMod, s_values_noisy, s_values, truncated_s_values_noisy
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')


def optDist(ins, rho_0,
            lamda, s_truns, sensitivity_delta, S_values):
    s_noisy = np.zeros(ins.m)
    s_trun = np.zeros(ins.m)
    opt_models = []
    s_val = []
    noisy_s_vals = []
    x_values = np.zeros(sum(ins.n))
    common_cap_duals = np.zeros((ins.m, ins.parties))
    private_cap_duals = np.zeros((sum(ins.m_parties), ins.parties))
    for i in range(ins.parties):
        optMod_tmp, s_tmp, s_values_real, truncated_s_tmp = optimize_by_party(ins.n[i],
                                                                              ins.r[sum(ins.n[0:i]):sum(ins.n[0:i+1])],
                                                                              ins.A[:, sum(ins.n[0:i]):sum(ins.n[0:i+1])],
                                                                              ins.B[sum(ins.m_parties[0:i]):sum(ins.m_parties[0:i+1]), 
                                                                                sum(ins.n[0:i]):sum(ins.n[0:i+1])],
                                                                              ins.c_individual[sum(ins.m_parties[0:i]):sum(ins.m_parties[0:i+1])],
                                                                              ins.product_demand[sum(ins.n[0:i]):sum(ins.n[0:i+1])],
                                                                              ins.c, rho_0,
                                                                              lamda,
                                                                              s_truns[i],
                                                                              sensitivity_delta,
                                                                              S_values[i])
        x_values[sum(ins.n[0:i]):sum(ins.n[0:i+1])] = [i.X for i in optMod_tmp.getVars()[:ins.n[i]]]
        common_cap_duals[:, i] = optMod_tmp.Pi[:ins.m]
        private_cap_duals[sum(ins.m_parties[0:i]):sum(ins.m_parties[0:i+1]), i] = optMod_tmp.Pi[3*ins.m:3*ins.m+ins.m_parties[i]]
        s_val.append(s_values_real)
        s_noisy = s_noisy + s_tmp
        s_trun = s_trun + truncated_s_tmp
        noisy_s_vals.append(truncated_s_tmp)
        opt_models.append(optMod_tmp)

    return (opt_models, s_noisy, s_trun, s_val, noisy_s_vals,
            x_values, common_cap_duals, private_cap_duals)


if __name__ == '__main__':
    optDist()
