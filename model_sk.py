# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:55:37 2019

@author: Utku Karaca
"""

from gurobipy import GRB, Model, GurobiError
import numpy as np

#   MAXIMIZE    sum r_k*x_k
#   subject to
#               A_k * x_k <= s_k
#               B_k * x_k <= c_k
#               sum s_k    =   c
#               x_k, s_k  >=   0


def optimize(n, m, parties, r, A, B, c, c_individual, upper_limit, demand):
    try:
        optMod = Model('Initial Model')
        x = optMod.addVars(int(sum(n)), lb = 0, name = 'x')
        s = optMod.addVars(m, parties, lb = 0, name = 's')
        obj_fn = sum(r[i]*x[i] for i in range(sum(n)))
        optMod.setObjective(obj_fn, GRB.MAXIMIZE)
        c_individual = np.concatenate((c_individual, demand))
        
        # A_k * x_k <= s_k
        for i in range(np.size(A, axis = 0)):
            for k in range(parties):
                optMod.addLConstr(sum(A[i,j]*x[j] for j in range(sum(n[0:k]), sum(n[0:k+1]))), 
                                    GRB.LESS_EQUAL,
                                    s[i,k],
                                    name = ('Shared Capacity Constraint for Source ' + 
                                            str(i+1) + ' and Party ' + str(k+1)))
        
        for i in range(np.size(A, axis = 0)):
            for k in range(parties):
                optMod.addLConstr(s[i, k], GRB.LESS_EQUAL, c[i],
                                  name = ('Auxiliary Shared Capacity Constraint for Source ' + 
                                          str(i+1)))
        
        for i in range(np.size(A, axis = 0)):
            for k in range(parties):
                optMod.addLConstr(s[i, k], GRB.LESS_EQUAL, upper_limit[k]*c[i],
                                  name = ('Upper Limit Shared Capacity Constraint for Source ' + 
                                          str(i+1) + ' for Party ' + str(k+1)))
        # B_k * x_k <= c_k        
        for i in range(np.size(B, axis = 0)):
            optMod.addLConstr(sum(B[i,j]*x[j] for j in range(np.size(B, axis = 1))), 
                              GRB.LESS_EQUAL, 
                              c_individual[i], 
                              name = 'Individual Capacity Constraint for Source ' + str(i+1))
        
        # sum s_k    =   c        
        for i in  range(np.size(A, axis = 0)):
            optMod.addLConstr(sum(s[i,k] for k in  range(parties)),
                            GRB.EQUAL,
                            c[i],
                            name = 'Portion of the Source ' + str(i+1))
        
        optMod.setParam('OutputFlag',False) # prevents output logs in the command line.
        optMod.optimize() # solves the model
        optMod.write("initial_model.lp") # outputs the LP file of the model
        return optMod
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')

if __name__ == '__main__':
    optimize()