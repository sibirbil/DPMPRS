# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:34:19 2019

@author: Utku Karaca
"""

from gurobipy import GRB, Model, GurobiError
import numpy as np

def optimize(n, r, A, B, c, c_individual, upper_limit, demand):
    try:
        B = np.concatenate((B, -1 * np.eye(sum(n))))
        c_individual = np.concatenate((c_individual, demand))
        optMod = Model('Model_1')
#        Decision Variables
        x = optMod.addVars(int(sum(n)), lb = 0, name = 'x')
#        Objective Function
        obj_fn = sum(r[i]*x[i] for i in range(sum(n)))
        optMod.setObjective(obj_fn, GRB.MAXIMIZE)
#        Constraints
        for i in range(np.size(A, axis = 0)):
            optMod.addLConstr(sum(A[i,j]*x[j] for j in range(np.size(A, axis = 1))), 
                              GRB.LESS_EQUAL, 
                              c[i], 
                              name='Capacity Constraint for Shared Source ' + str(i+1))
        
        for i in range(np.size(A, axis = 0)):
            for j in range(len(n)):
                tempA = A[i,sum(n[0:j]):(sum(n[0:j]) + n[j])]
                optMod.addLConstr(sum(tempA[k] * x[(sum(n[0:j]) + k)] for k in range(n[j])),
                                  GRB.LESS_EQUAL,
                                  upper_limit[j]*c[i],
                                  name=('Upper Bound on Shared Capacity for Source ' + str(i+1) + 
                                        ' for Party ' + str(j+1)))
        
        for i in range(np.size(B, axis = 0)):
            optMod.addLConstr(sum(B[i,j]*x[j] for j in range(np.size(B, axis = 1))), 
                              GRB.LESS_EQUAL, 
                              c_individual[i], 
                              name='Individual Capacity Constraint for Source ' + str(i+1))
#        Solve the model
        optMod.setParam('OutputFlag',False) # Prevent output logs in the command line.
        optMod.optimize() # solve the model
        optMod.write("first_model.lp") # output the LP file of the model
        return optMod
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')

if __name__ == '__main__':
    optimize()