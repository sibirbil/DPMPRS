# Differential Privacy in Multi-Party Resource Sharing

Utku Karaca, İlker Birbil, Nurşen Aydiın and Gizem Mullaoğlu

## Abstract

This study examines a resource-sharing problem involving multiple
parties that agree to use a set of capacities together. We start with
modeling the whole problem as a mathematical program, where all
parties are required to exchange information to obtain the optimal
objective function value. This information bears private data from
each party in terms of coefficients used in the mathematical
program. Moreover, the parties also consider the individual optimal
solutions as private. In this setting, the concern for the parties is
the privacy of their data and their optimal allocations. We propose a
two-step approach to meet the privacy requirements of the parties. In
the first step, we obtain a reformulated model that is amenable to a
decomposition scheme. Although this scheme eliminates almost all data
exchange, it does not provide a formal privacy guarantee. In the
second step, we provide this guarantee with a differentially private
algorithm at the expense of deviating slightly from the optimality. We
provide bounds on this deviation and discuss the consequences of these
theoretical results. The study ends with a simulation study on a
planning problem that demonstrates an application of the proposed
approach. Our work provides a new optimization model and a solution
approach for optimal allocation of a set of shared resources among
multiple parties who expect privacy of their data. The proposed
approach is based on the decomposition of the shared resources and the
randomization of the optimization iterations. With our analysis, we
show that the resulting randomized algorithm does give a guarantee for
the privacy of each party’s data. As we work with a general
optimization model, our analysis and discussion can be used in
different application areas including production planning, logistics,
and network revenue management.

**Keywords:** collaboration; differential privacy; resource sharing; decomposition

## Simulation Study

One can run test.py to start the simulation study. The results will appear in the folder "runResults."

Required packages: numpy, csv, gurobipy

Required solver: Gurobi

For each scenario, there will be four actions.
1. Two txt files will be created with names showing the dynamic parameters (can be seen below). The files include the following:
- Optimal objective function values
- Data private model iteration results (in an every 10 iterations)
- Data private model objective function value
- Data private model decision variable values
- Differentially private model iteration results (in an every 10 iterations)
- Differentially private model objective function value
- Differentially private model decision variable values

2. For each scenario, a row will be added to "results.csv". The row includes the following:
- the parameters of the scenario
- objective function values of 3 models
- primal solutions of 3 models
  - (parties * max_product) entries for the initial model
  - (parties * max_product) entries for the distributed model
  - (parties * max_product) entries for the differentially private model
- dual solutions of 3 models
  - (m + parties * max_private_cap) entries for the initial model
  - parties * (m + max_private_cap) entries for the distributed model
  - parties * (m + max_private_cap) entries for the differentially private model

3. For each scenario, a row will be added to both dataPrivate.csv and
diffPrivate.csv. Rows include the following:
- dataPrivate.csv
  - seed value
  - \bar{s}_k values for each party
  -  best objective function value so far
- diffPrivate.csv
  - seed value
  - \bar{s}_k values for each party
  - shared resource capacities
  - best s_k values
  -  epsilon
  -  delta
  - best objective function value so far
		
Current dynamic parameters for the simulation in test.py
- Number of parties (parties): 5  
- Number of shared resources (m): 5
- Epsilon (epsilon): [0.05, 0.10, 0.15, 0.20, 0.25]
- Delta (delta): [0.05, 0.10, 0.15, 0.20]
- Upper Limit on shared resources (upper_limit_policy, upper_limit_coef): [0.15, 0.30, 0.50, 1.00] 
- Seed (seed): 1 to 100 with increments 1

Current static parameters for the simulation in test.py
- maximum number of iterations for data-private model (max_iter_subgradient): 5000
- maximum number of iterations for differentially private model (max_iter): 100
- maximum number of product for each party (max_product): 15
- maximum number of private capacity for each party (max_private_cap): 10

The inputs of the original model (A_k, B_k, b_k, r_k, c) are mostly generated randomly. The parameters of these inputs can be seen in Section 3 of the paper.
