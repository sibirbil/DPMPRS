# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:32:02 2021

@author: Utku Karaca
"""

def data_creation(parties, m, max_private_cap, max_product, seed):
    import numpy as np
    from numpy import linalg as LA
    np.random.seed(seed)
# Parameters
    # number of individual capacities
    m_parties = np.random.randint(low = 5, 
                                  high = max_private_cap + 1, 
                                  size = m)     
    
    # number of products offered by parties
    n = np.random.randint(low = 10, 
                          high = max_product +1 , 
                          size = m)
    
# Coefficients for revenue and capacity vectors 
    revenue_coef = 100          # each revenue vector (r_k) will be multiplied with this
    capacity_coef = 10          # each capacity vector (both c, c_k) will be multiplied with this

# Random Generation of 
    #   A (made of A_k's), 
    #   B (made of B_k's), 
    #   r (made of r_k's), 
    #   c_individual (made of c_k's), 
    #   c
    
    r = 50 + revenue_coef * np.random.rand(sum(n))
    A = 5 * np.random.rand(m, sum(n))
    B = np.zeros((sum(m_parties), sum(n)))
    for i in range(parties):
        party_matrix = np.zeros((m_parties[i], n[i]))
        if i == 0:
            party_matrix[0:m_parties[i], 0:n[i]] = np.random.rand(m_parties[i], n[i])
            B[0:m_parties[i], 0:n[i]] = party_matrix
        else:
            party_matrix[0:m_parties[i], 0:n[i]] = np.random.rand(m_parties[i],n[i])
            B[sum(m_parties[0:i]):(sum(m_parties[0:i]) + m_parties[i]), 
              sum(n[0:i]):(sum(n[0:i]) + n[i])] = party_matrix
    
        
    c = 10 + capacity_coef * np.random.rand(m)
    c_individual = 10 + capacity_coef * np.random.rand(sum(m_parties))
    
    # Compute norm of c and divide A by norm(c)
    c = c / LA.norm(c)
    A = A / LA.norm(c)
    
    
    return n, m_parties, r, A, B, c, c_individual


def init_model(m, parties, n, 
               r, A, B, c, c_individual, upper_limit_coef):
    import numpy as np
    import model_original
    import model_sk # Model with s_k introduced
    
# Coefficient for demand vectors 
    demand_coef = 0.25          # each products' demand (d_k) will be multiplied with this
    
# Upper Limit creation for each party
    if np.size(upper_limit_coef) > 1:
        upper_limit = upper_limit_coef
    elif np.size(upper_limit_coef) == 1:
        upper_limit = upper_limit_coef * np.ones(parties)
 # Original model
    original_model = model_original.optimize(n, r, A, B, c, c_individual, upper_limit, np.zeros(sum(n))) 
    
    # get optimal values of x's
    optimal_x = np.zeros(sum(n))
    for i in range(sum(n)):
        optimal_x[i] = original_model.getVars()[i].x

# Create demand vector
    product_demand = np.zeros(sum(n))
    for i in range(sum(n)):
        if optimal_x[i] > 0:
            product_demand[i] = optimal_x[i] * (1 - demand_coef * np.random.random())
    
    product_demand = product_demand * -1
# Solve the original model with demand added
    original_model_demand = model_original.optimize(n, r, A, B, c, c_individual, upper_limit, product_demand)
    
    
# Initial model where everybody input their information to the model, also s_k is introduced
    init_mod = model_sk.optimize(n, int(m), int(parties), r, A, B, c, c_individual, upper_limit, product_demand)
    
    return original_model, original_model_demand, init_mod, product_demand


def dataPrivate_model(m, parties, m_parties, n, 
                      r, A, B, c, c_individual, 
                      product_demand, step_len,
                      init_mod, upper_limit_coef, 
                      f):
    import numpy as np
    import model_distr
# Coefficients
    max_iter_subgradient = 5000
# Upper Limit creation for each party
    if np.size(upper_limit_coef) > 1:
        upper_limit = upper_limit_coef
    elif np.size(upper_limit_coef) == 1:
        upper_limit = upper_limit_coef * np.ones(parties)
# Subgradient algorithm
    # lambda is initialized as 0 vector
    # while not coverged
    #   each party solves its subproblem, returns s_k
    #   if better solution obtained, then it is stored
    #   lambda values are updated
    lamda = np.median(r)*np.zeros(m)
    iteration = 1
    best_obj_vals = []
    best_obj_val = init_mod.objVal * 1000
    f.write('2nd Mdl Obj\t|Dist Mdl\tDist Min\t#Iter\n')
    f.write('--------------------------------------------------------------------------------\n')
    while iteration < max_iter_subgradient:
        # Solves subproblems and returns the summation of s_k's and each submodels
        (opt_mod, s_total, x_values, 
         private_cap_duals, 
         common_cap_duals) = model_distr.optDist(n,
                                                 int(m), 
                                                 m_parties, 
                                                 int(parties), 
                                                 r, A, B, c, c_individual, upper_limit,
                                                 product_demand,
                                                 lamda)
        
        # Keeps the best solution until now
        if sum(opt_mod[i].objVal for i in range(parties)) + np.inner(c, lamda) < best_obj_val:
            best_obj_val = sum(opt_mod[i].objVal for i in range(parties)) + np.inner(c, lamda)
            opt_mod_min = opt_mod
            min_iter = iteration
            best_s_total = s_total
            best_lamda = lamda
            best_x_val = x_values
            best_private_cap_duals = private_cap_duals
            best_common_cap_duals = common_cap_duals
        best_obj_vals.append(best_obj_val)
        # Prints the current situation in an every 10 iteration
        if iteration % 10 == 0 or iteration == 1:
            f.write('%.5f\t|%.5f\t%.5f\t%d\n' % 
                    (init_mod.objVal, 
                     sum(opt_mod[i].objVal for i in range(parties)) + np.inner(c, lamda), 
                     best_obj_val, 
                     min_iter,
                     ))
        
        # Lambda updates
        gradient = s_total - c
        if max(abs(gradient)) <= 1.0e-6:
            break
        if step_len == 1:
            nu = 1.0/(iteration**(0.5))
        elif step_len == 2:
            nu = 1.0/(iteration**(0.51))
            
        lamda = np.maximum(0, lamda + nu * gradient)
        ###
        iteration = iteration + 1
    
    f.write('--------------------------------------------------------------------------------\n')
    f.write('Obj Value Comparison(Init Model and Distributed Model)\n')
    f.write('--------------------------------------------------------------------------------\n')
    f.write('%f\t%f\n' % (init_mod.objVal, best_obj_val))
    f.write('--------------------------------------------------------------------------------\n')
    f.write('Primal Variables Comparison(Init Model and Distributed Model)\n')
    f.write('--------------------------------------------------------------------------------\n')
    
    f.write('--- Initial Model ---\n')
    for v in init_mod.getVars():
        if v.x >= 0:
            f.write('%s %g\n' % (v.varName, v.x))
    f.write('--- Second Model ---\n')
    for iteration, i in enumerate(opt_mod_min):
        f.write('Party ' + str(iteration+1) + '\n')
        for v in i.getVars():
            if v.x >= 0:
                f.write('%s %g\n' % (v.varName, v.x))
                
    return best_obj_vals, best_obj_val, best_x_val, best_private_cap_duals, best_common_cap_duals


def diffPrivate_model(m, parties, m_parties, n, 
                      r, A, B, c, c_individual, 
                      product_demand, step_len,
                      epsilon, delta, init_mod,
                      upper_limit_coef,
                      f):
    import numpy as np
    import model_dp
    max_iter = 100              # number of iterations that the differentially private algorithm
                                    #   is ran
# Upper Limit creation for each party
    if np.size(upper_limit_coef) > 1:
        upper_limit = upper_limit_coef
    elif np.size(upper_limit_coef) == 1:
        upper_limit = upper_limit_coef * np.ones(parties)
# Differentially private subgradient algorithm
    # lambda is initialized as 0 vector
    # while not coverged
    #   each party solves its subproblem, returns noisy s_k's (and also normal s_k)
    #   if better solution obtained (using noisy s_k's), then it is stored
    #   lambda values are updated
    lamda_dp = np.median(r)*np.zeros(m)
    iteration = 1
    best_obj_vals = []
    best_obj_val_dp = init_mod.objVal * 1000
    f.write('1st Mdl Obj\t|Prvt Mdl\tPrvt Min\t#Iter\n')
    f.write('--------------------------------------------------------------------------------\n')
    while iteration < max_iter:
        # Solves subproblems and returns the summation
        #   of noisy and normal s_k's and each submodels
        (opt_mod_dp, s_total_noisy, s_total_real, x_val_dp,
         common_cap_duals_dp, private_cap_duals_dp) = model_dp.optDist(n, 
                                                                       int(m), 
                                                                       m_parties, 
                                                                       int(parties), 
                                                                       r, A, B, c, c_individual,
                                                                       upper_limit,
                                                                       product_demand,
                                                                       lamda_dp, 
                                                                       epsilon, delta, max_iter)
        # Keeps the best solution until now
        current_obj_val_dp = (sum(opt_mod_dp[i].objVal for i in range(parties)) + 
                              np.inner(c, lamda_dp))
        
        if current_obj_val_dp < best_obj_val_dp:
            best_obj_val_dp = current_obj_val_dp
            opt_mod_dp_min = opt_mod_dp
            min_iter_dp = iteration
            best_s_total_real = s_total_real
            best_x_val_dp = x_val_dp
            best_private_cap_duals_dp = private_cap_duals_dp
            best_common_cap_duals_dp = common_cap_duals_dp
        best_obj_vals.append(best_obj_val_dp)
        # Prints the current situation in an every 10 iteration
        if iteration % 10 == 0 or iteration == 1:
            f.write('%.5f\t|%.5f\t%.5f\t%d\n' % 
                    (init_mod.objVal, 
                     current_obj_val_dp,
                     best_obj_val_dp, 
                     min_iter_dp
                     ))
        
        # Lambda updates
        gradient_dp = s_total_noisy - c
        if max(abs(gradient_dp)) <= 1.0e-6:
            break
        if step_len == 1:
            nu = 1.0/(iteration**(0.5))
        elif step_len == 2:
            nu = 1.0/(iteration**(0.51))
        lamda_dp = np.maximum(0, lamda_dp + nu * gradient_dp)
        
        # Increase the iteration number
        iteration = iteration + 1
    
    # Real usage on the last iteration and on the best solution in the DP Algorithm
    ss_best_total = []
    for j in range(len(c)):
        s = 0
        for i in range(parties):
            s = s + s_total_real[i][j]
        ss_best_total.append(s)
    
    ss_best_total_real = []
    for j in range(len(c)):
        s = 0
        for i in range(parties):
            s = s+best_s_total_real[i][j]
        ss_best_total_real.append(s)
    
    f.write('--------------------------------------------------------------------------------\n')
    f.write('Obj Value Comparison(1st Model and 2nd Model)\n')
    f.write('--------------------------------------------------------------------------------\n')
    f.write('%f\t%f\n' % (init_mod.objVal, best_obj_val_dp))
    f.write('--------------------------------------------------------------------------------\n')
    f.write('Primal Variables Comparison(1st Model, 2nd Model and 3rd Model)\n')
    f.write('--------------------------------------------------------------------------------\n')
    
    f.write('--- Initial Model ---\n')
    for v in init_mod.getVars():
        if v.x > 0:
            f.write('%s %g\n' % (v.varName, v.x))
    f.write('--- Second Model ---\n')
    for iteration, i in enumerate(opt_mod_dp_min):
        f.write('Party ' + str(iteration+1) + '\n')
        for v in i.getVars():
            if v.x > 0:
                f.write('%s %g\n' % (v.varName, v.x))
    
    return best_obj_vals, best_obj_val_dp, best_x_val_dp, best_private_cap_duals_dp, best_common_cap_duals_dp, s_total_real


def result_row(parties, m, n, m_parties, 
               max_private_cap, max_product,
               init_mod,
               best_obj_val, best_x_val, best_obj_val_dp, best_x_val_dp,
               best_common_cap_duals, best_private_cap_duals,
               best_common_cap_duals_dp, best_private_cap_duals_dp):
    
    # Returned (entries)
            #    - objective function values of 3 models
            #    - primal solutions of 3 models
            #        - (parties * max_product) entries for the initial model
            #        - (parties * max_product) entries for the distributed model
            #        - (parties * max_product) entries for the differentially private model
            #    - dual solutions of 3 models
            #        - (m + parties * max_private_cap) entries for the initial model
            #        - parties * (m + max_private_cap) entries for the distributed model
            #        - parties * (m + max_private_cap) entries for the differentially private model
    import numpy as np
    
    returned = -2*np.ones(3+3*parties*max_product + 
                                  m + parties * max_private_cap + 
                                  parties * (m + max_private_cap) + 
                                  parties * (m + max_private_cap))
    # recording objective function values
    returned[0] = init_mod.objVal   # initial problem objective function value
    returned[1] = best_obj_val            # distributed problem objective function value
    returned[2] = best_obj_val_dp         # differentially private problem objective function value
    
    # recording x_k variable values
    returned[3:(3+sum(n))] = [i.X for i in init_mod.getVars()[:sum(n)]]                     # initial problem x_k values
    returned[(3 + parties*max_product):(3+parties*max_product+sum(n))] = best_x_val         # distributed problem x_k values
    returned[(3 + 2*parties*max_product):(3+2*parties*max_product+sum(n))] = best_x_val_dp  # differentially private problemx_k values
    
    # index update
    index = 3 + 3*parties*max_product
    
    # recording dual variable values of the initial model (first shared, then private constraints)
    duals_optimal = -1 * np.ones(m + parties * max_private_cap)
    
    duals_optimal[:m] = init_mod.Pi[-m:]
    private_cap_dual_optimal = init_mod.Pi[(3*m*parties):(3*m*parties+sum(m_parties))]
    
    for i in range(parties):
        temp = np.array(private_cap_dual_optimal[sum(m_parties[0:i]):sum(m_parties[0:i+1])])
        if len(temp) < max_private_cap:
            temp = np.append(temp, -1*np.ones(max_private_cap-len(temp)))
        
        duals_optimal[(m + i * max_private_cap):(m + (i + 1) * max_private_cap)] = temp
        
    returned[index:(index + m + parties * max_private_cap)] = duals_optimal
    
    # recording dual variable values of the distributed model
    # index update
    index = index + m + parties * max_private_cap
    #   note that each party will have maximum of (m+10) dual variables
    for i in range(parties):
        temp = np.concatenate((best_common_cap_duals[:,i],
                               best_private_cap_duals[sum(m_parties[0:i]):sum(m_parties[0:i+1]),i]))
        if len(temp) < m+max_private_cap:
            temp = np.concatenate((temp, -1*np.ones(m+max_private_cap-len(temp))))
        returned[(index + i * (m + max_private_cap)):(index + (i + 1) * (m + max_private_cap))] = temp
        
        
    # index update
    index = index + parties * (m + max_private_cap)
    
    # recording dual variable values of the differentially private model
    for i in range(parties):
        temp = np.concatenate((best_common_cap_duals_dp[:,i],
                               best_private_cap_duals_dp[sum(m_parties[0:i]):sum(m_parties[0:i+1]),i]))
        if len(temp) < m+max_private_cap:
            temp = np.concatenate((temp, -1*np.ones(m+max_private_cap-len(temp))))
        returned[(index + i * (m + max_private_cap)):(index + (i + 1) * (m + max_private_cap))] = temp
        
    return returned


def simulation(parameter_list):
    import numpy as np
    import csv
    parties = int(parameter_list[0])
    m = int(parameter_list[1])
    step_len = int(parameter_list[2])
    upper_limit_policy = parameter_list[3]
    seed = int(parameter_list[4])
    max_private_cap = 10
    max_product = 15
    
    name = ('parties_' + str(parties) +
            '_sharedCap_' + str(m) +
            '_skBoundPolicy_' + '%.0f' % int(upper_limit_policy) +
            '_seed_' + str(seed) + '.txt')
    # Create data
    n, m_parties, r, A, B, c, c_individual = data_creation(parties, m, 
                                                           max_private_cap, 
                                                           max_product, 
                                                           seed)
    
    # Upper limit policy generation
    total_usage = 1.2
    random_numbers = np.random.rand(parties - 1)
    random_numbers = random_numbers / np.sum(random_numbers)
    if upper_limit_policy == 1:         # low-share policy
        random_numbers = random_numbers * (total_usage - 0.15)
        upper_limit_coef = [0.15]
        upper_limit_coef = np.append(upper_limit_coef, random_numbers)
    elif upper_limit_policy == 2:       # mid-share policy
        random_numbers = random_numbers * (total_usage - 0.30)
        upper_limit_coef = [0.30]
        upper_limit_coef = np.append(upper_limit_coef, random_numbers)
    elif upper_limit_policy == 3:                               # high-share policy
        random_numbers = random_numbers * (total_usage - 0.50)
        upper_limit_coef = [0.50]
        upper_limit_coef = np.append(upper_limit_coef, random_numbers)
    elif upper_limit_policy == 4:
        upper_limit_coef = [1,1,1,1,1]
    # Solve initial model with and without demand
    (original_model, 
     original_model_demand, 
     init_mod, 
     product_demand) = init_model(m, parties, n, r, A, B, c, c_individual, upper_limit_coef)
    
    f = open('./runResults/'+name, "w+")
    f.write('Objective value of the model without demand:\t' + str(original_model.objVal) +  
            '\nObjective value of the model with demand:   \t' + str(original_model_demand.objVal) + '\n')
    
    # Initial model where everybody input their information to the model, also s_k is introduced
    f.write('\t\tModel with additional constraints (s_k included model)\n')
    f.write('------------------------------------------------------------------------------------\n')
    f.write('Obj:\t' + str(init_mod.objVal) + '\n')
    
    
    # Solve data private model
    (obj_vals,
     best_obj_val, 
     best_x_val, 
     best_private_cap_duals, 
     best_common_cap_duals) = dataPrivate_model(m, parties, m_parties, n, 
                                                r, A, B, c, c_individual, 
                                                product_demand, step_len,
                                                init_mod, upper_limit_coef,
                                                f)
    f.close()
    obj_vals.insert(0, seed)
    obj_vals.insert(1, upper_limit_coef)
    with open('./runResults/dataPrivate.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(obj_vals)                                          
    
    # Solve differentially private model for all epsilon and delta values
    for epsilon in [0.05,0.10,0.15,0.20,0.25]:
        for delta in [0.05,0.10,0.15,0.20]:
            name = ('parties_' + str(parties) +
                    '_sharedCap_' + str(m) +
                    '_skBoundPolicy_' + '%.0f' % upper_limit_policy +
                    '_seed_' + str(seed) + 
                    '_eps_' + '%.2f' % epsilon +
                    '_delta_' + '%.2f' % delta +'.txt')
            f = open('./runResults/'+name, "w+")
            (obj_vals_dp, 
             best_obj_val_dp, 
             best_x_val_dp, 
             best_private_cap_duals_dp, 
             best_common_cap_duals_dp, 
             s_total_real) = diffPrivate_model(m, parties, m_parties, n, 
                                               r, A, B, c, c_individual, 
                                               product_demand, step_len,
                                               epsilon, delta, init_mod,
                                               upper_limit_coef,
                                               f)
            f.close()
            obj_vals_dp.insert(0, seed)
            obj_vals_dp.insert(1, upper_limit_coef)
            obj_vals_dp.insert(2, c)
            obj_vals_dp.insert(3, s_total_real[0])
            obj_vals_dp.insert(4, s_total_real[1])
            obj_vals_dp.insert(5, s_total_real[2])
            obj_vals_dp.insert(6, s_total_real[3])
            obj_vals_dp.insert(7, s_total_real[4])
            obj_vals_dp.insert(8, epsilon)
            obj_vals_dp.insert(9, delta)
            with open('./runResults/diffPrivate.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(obj_vals_dp)
            returned = result_row(parties, m, n, m_parties, 
                       max_private_cap, max_product,
                       init_mod,
                       best_obj_val, best_x_val, best_obj_val_dp, best_x_val_dp,
                       best_common_cap_duals, best_private_cap_duals,
                       best_common_cap_duals_dp, best_private_cap_duals_dp)
    
            a = np.hstack(((parties, m, epsilon, delta, upper_limit_coef, seed, returned)))
    
            with open('./runResults/results.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(a)
    
    return 1


if __name__=="__main__":
    from tools import cartesian
    grid = list(cartesian(([5],             # number of parties
                           [5],             # number of shared capacities
                           [1],             # 1: step_len nu^(0.50), 2: step_len nu^(0.51)
                           [1,2,3,4],       # upper limit scenarios for shared capacity usage, 
                                            # low, mid, high [0.15, 0.30, 0.50, 1.00]
                           range(1,101))))  # seed
    
    for i in range(len(grid)):
        simulation(grid[i])
    
    

    