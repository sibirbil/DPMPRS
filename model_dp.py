# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:54:20 2020

@author: Utku Karaca
"""

from gurobipy import GRB, Model, GurobiError
import numpy as np


# Each party solves the following problem (optimize_by_party)
#
#   MAXIMIZE    r_k*x_k
#   subject to
#               A_k * x_k <= s_k
#               B_k * x_k <= c_k
#               x_k, s_k  >=   0
#
# Returns the vector s_k and noisy s_k

def optimize_by_party(n_ind, r_ind, A_ind, B_ind, c_ind, demand_ind, c, upper_limit, lamda, epsilon, delta, max_iter):
    try:
        optMod = Model('Model_5')
        x = optMod.addVars(int(n_ind), lb = 0, name = 'x')
        s = optMod.addVars(len(lamda), lb = 0, name = 's')
        obj_fn = (sum(r_ind[i]*x[i] for i in range(n_ind)) -
                  sum(lamda[i]*s[i] for i in range(len(lamda)))
                  )
        optMod.setObjective(obj_fn, GRB.MAXIMIZE)
        
        for i in range(len(lamda)):
            optMod.addLConstr(sum(A_ind[i,j]*x[j] for j in range(n_ind)), 
                             GRB.LESS_EQUAL,
                             s[i],
                             name = ('Shared Capacity Constraint for Source ' + str(i+1))
                             )
        
        for i in range(len(lamda)):
            optMod.addLConstr(s[i], GRB.LESS_EQUAL, c[i],
                              name = ('Auxiliary Shared Capacity Constraint for Source ' + str(i+1)))
        
        for i in range(len(lamda)):
            optMod.addLConstr(s[i], GRB.LESS_EQUAL, upper_limit * c[i],
                              name = ('Upper Bound on Shared Capacity Constraint for Source ' + str(i+1)))
        
        for i in range(len(c_ind)):
            optMod.addLConstr(sum(B_ind[i,j]*x[j] for j in range(n_ind)), 
                              GRB.LESS_EQUAL, 
                              c_ind[i], 
                              name = ('Individual Capacity Constraint for Source ' + 
                                      str(i+1)))
        
        for i in range(len(demand_ind)):
            optMod.addLConstr(-1 * x[i], GRB.LESS_EQUAL, demand_ind[i],
                              name = ('Demand for Product '+ str(i+1)))
        
        optMod.setParam('OutputFlag',False) # Prevent output logs in the command line.
        optMod.optimize() # solve the model
        optMod.write('dp_model.lp')
        s_values = np.array(list(s[i].x for i in range(len(s))))
        
        ## Add noise to the new s_values
        noise = upper_limit * np.random.laplace(loc = 0, 
                                                scale = np.sqrt(((4*max_iter)/epsilon**2) * 
                                                                np.log(np.e+(epsilon/delta))),
                                                size = len(s_values))
        # noise = upper_limit * np.random.normal(loc = 0, 
        #                                         scale = np.sqrt(((8*max_iter)/epsilon**2) * 
        #                                                         np.log(np.e+(epsilon/delta))),
        #                                         size = len(s_values))
        s_values_noisy = s_values + noise
        
        return optMod, s_values_noisy, s_values
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')

def optDist(n, m, m_parties, parties, r, A, B, c, c_individual, upper_limit, demand, lamda, epsilon, delta, max_iter):
    s_noisy = np.zeros(m)
    opt_models = []
    s_val = []
    x_values = np.zeros(sum(n))
    common_cap_duals = np.zeros((m,parties))
    private_cap_duals = np.zeros((sum(m_parties),parties))
    for i in range(parties):
        optMod_tmp, s_tmp, s_values_real = optimize_by_party(n[i], 
                                                             r[sum(n[0:i]):sum(n[0:i+1])],
                                                             A[:,sum(n[0:i]):sum(n[0:i+1])],
                                                             B[sum(m_parties[0:i]):sum(m_parties[0:i+1]), 
                                                               sum(n[0:i]):sum(n[0:i+1])],
                                                             c_individual[sum(m_parties[0:i]):sum(m_parties[0:i+1])],
                                                             demand[sum(n[0:i]):sum(n[0:i+1])],
                                                             c, upper_limit[i],
                                                             lamda, 
                                                             epsilon, delta, max_iter)
        x_values[sum(n[0:i]):sum(n[0:i+1])] = [i.X for i in optMod_tmp.getVars()[:n[i]]]
        common_cap_duals[:,i] = optMod_tmp.Pi[:m]
        private_cap_duals[sum(m_parties[0:i]):sum(m_parties[0:i+1]),i] = optMod_tmp.Pi[3*m:3*m+m_parties[i]]
        s_val.append(s_values_real)
        s_noisy = s_noisy + s_tmp
        opt_models.append(optMod_tmp)
    return opt_models, s_noisy, s_val, x_values, common_cap_duals, private_cap_duals

if __name__ == '__main__':
    optDist()