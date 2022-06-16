# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:34:19 2019

@author: Utku Karaca
"""

from gurobipy import GRB, Model, GurobiError
import numpy as np


def optimize(ins, demand):
    try:
        ins.B = np.concatenate((ins.B, -1 * np.eye(sum(ins.n))))
        ins.c_individual = np.concatenate((ins.c_individual, demand))
        optMod = Model('Model_1')
#        Decision Variables
        x = optMod.addVars(int(sum(ins.n)), lb=0, name='x')
#        Objective Function
        obj_fn = sum(ins.r[i]*x[i] for i in range(sum(ins.n)))
        optMod.setObjective(obj_fn, GRB.MAXIMIZE)
#        Constraints
        for i in range(np.size(ins.A, axis=0)):
            optMod.addLConstr(sum(ins.A[i, j]*x[j] for j in range(np.size(ins.A, axis=1))),
                              GRB.LESS_EQUAL,
                              ins.c[i],
                              name='Capacity Constraint for Shared Source ' + str(i+1))

        # for i in range(np.size(A, axis = 0)):
        #     for j in range(len(n)):
        #         tempA = A[i,sum(n[0:j]):(sum(n[0:j]) + n[j])]
        #         optMod.addLConstr(sum(tempA[k] * x[(sum(n[0:j]) + k)] for k in range(n[j])),
        #                           GRB.LESS_EQUAL,
        #                           upper_limit[j]*c[i],
        #                           name=('Upper Bound on Shared Capacity for Source ' + str(i+1) +
        #                                 ' for Party ' + str(j+1)))

        for i in range(np.size(ins.B, axis=0)):
            optMod.addLConstr(sum(ins.B[i, j]*x[j] for j in range(np.size(ins.B, axis=1))),
                              GRB.LESS_EQUAL,
                              ins.c_individual[i],
                              name='Individual Capacity Constraint for Source ' + str(i+1))
#        Solve the model
        optMod.setParam('OutputFlag', False)  # Prevent output logs in the command line.
        optMod.optimize()  # solve the model
        optMod.write("first_model.lp")  # output the LP file of the model
        return optMod
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')


if __name__ == '__main__':
    optimize()
