# diferentiallyPrivateDecisionMaking
This repo includes codes for our recent publication.

One can run simulate.py to start the simulation study. The results will appear in the folder "results".

Required packages: numpy, csv, gurobipy

Required solver: Gurobi

For each scenario, there will be four actions.
- Two txt files will be created with names showing the dynamic parameters (can be seen below). The files include the following
    - Optimal objective function values
	- Data private model iteration results (in an every 10 iterations)
	- Data private model objective function value
	- Data private model decision variable values
	- Differentially private model iteration results (in an every 10 iterations)
	- Differentially private model objective function value
	- Differentially private model decision variable values
- For each scenario, a row will be added to "results.csv". The row includes the following
    - the parameters of the scenario
	- objective function values of 3 models
    - primal solutions of 3 models
	    - \# (parties \* max_product) entries for the initial model
        - \# (parties \* max_product) entries for the distributed model
        - \# (parties \* max_product) entries for the differentially private model
    - dual solutions of 3 models
        - \# (m + parties \* max_private_cap) entries for the initial model
        - \# parties \* (m + max_private_cap) entries for the distributed model
        - \# parties \* (m + max_private_cap) entries for the differentially private model
	3) For each scenario, a row will be added to both dataPrivate.csv and diffPrivate.csv. Rows include the followings
		- dataPrivate.csv
			- seed value
			- \bar{s}_k values for each party
		    - best objective function value so far
		- diffPrivate.csv
			- seed value
			- \bar{s}_k values for each party
			- shared resource capacities
			- best s_k values
			- epsilon
			- delta
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