# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:00:44 2021

Updated on Mon Jun 13 11:00:00 2022

@author: Utku Karaca
"""


def init_model(ins):
    import numpy as np
    import codes.model_original as model_original
    import codes.model_sk as model_sk             # Model with s_k introduced
# Coefficient for demand vectors
    demand_coef = 0.25          # each products' demand (d_k) will be multiplied with this
# Upper Limit creation for each party
    # if np.size(upper_limit_coef) > 1:
    #     upper_limit = upper_limit_coef
    # elif np.size(upper_limit_coef) == 1:
    #     upper_limit = upper_limit_coef * np.ones(parties)
# Original model
    original_model = model_original.optimize(ins, np.zeros(sum(ins.n)))
    # get optimal values of x's
    optimal_x = np.zeros(sum(ins.n))
    for i in range(sum(ins.n)):
        optimal_x[i] = original_model.getVars()[i].x

# Create demand vector
    product_demand = np.zeros(sum(ins.n))
    for i in range(sum(ins.n)):
        if optimal_x[i] > 0:
            product_demand[i] = optimal_x[i] * (1 - demand_coef * np.random.random())

    product_demand = product_demand * -1
# Solve the original model with demand added
    original_model_demand = model_original.optimize(ins, product_demand)


# Initial model where everybody input their information to the model, also s_k is introduced
    init_mod = model_sk.optimize(ins, product_demand)

    return original_model, original_model_demand, init_mod, product_demand


def dataPrivate_model(ins, step_len, init_mod, f, heavy_ball, beta):
    import numpy as np
    import codes.model_distr as model_distr
# Coefficients
    max_iter_subgradient = 10
# Upper Limit creation for each party
    # if np.size(upper_limit_coef) > 1:
    #     upper_limit = upper_limit_coef
    # elif np.size(upper_limit_coef) == 1:
    #     upper_limit = upper_limit_coef * np.ones(parties)
# Subgradient algorithm
    # lambda is initialized as 0 vector
    # while not coverged
    #   each party solves its subproblem, returns s_k
    #   if better solution obtained, then it is stored
    #   lambda values are updated
    lamda = np.median(ins.r)*np.zeros(ins.m)
    lamda_vals = []
    lamda_vals.append(lamda)
    iteration = 1
    best_obj_vals = []
    best_obj_val = init_mod.objVal * 1000
    if heavy_ball:
        f.write('Heavy Ball is used with beta: '+str(beta)+'\n')
    f.write('2nd Mdl Obj\t|Dist Mdl\tDist Min\t#Iter\n')
    f.write('--------------------------------------------------------------------------------\n')
    while iteration < max_iter_subgradient:
        # Solves subproblems and returns the summation of s_k's and each submodels
        (opt_mod, s_total, x_values,
         private_cap_duals,
         common_cap_duals) = model_distr.optDist(ins, lamda)

        # Keeps the best solution until now
        if sum(opt_mod[i].objVal for i in range(ins.parties)) + np.inner(ins.c, lamda) < best_obj_val:
            best_obj_val = sum(opt_mod[i].objVal for i in range(ins.parties)) + np.inner(ins.c, lamda)
            opt_mod_min = opt_mod
            min_iter = iteration
            # best_s_total = s_total
            # best_lamda = lamda
            best_x_val = x_values
            best_private_cap_duals = private_cap_duals
            best_common_cap_duals = common_cap_duals
        best_obj_vals.append(best_obj_val)
        # Prints the current situation in an every 10 iteration
        if iteration % 10 == 0 or iteration == 1:
            f.write('%.5f\t|%.5f\t%.5f\t%d\n' %
                    (init_mod.objVal,
                     sum(opt_mod[i].objVal for i in range(ins.parties)) + np.inner(ins.c, lamda),
                     best_obj_val,
                     min_iter,
                     ))

        # Lambda updates
        gradient = s_total - ins.c
        if max(abs(gradient)) <= 1.0e-6:
            break

        if step_len == 1:
            nu = 1.0/(iteration**(0.5))
        elif step_len == 2:
            nu = 1.0/(iteration**(0.51))

        if heavy_ball and iteration > 2:
            lamda = np.maximum(0, lamda + nu * gradient + beta * (lamda - lamda_vals[-2]))
        else:
            lamda = np.maximum(0, lamda + nu * gradient)
        ###
        iteration = iteration + 1
        lamda_vals.append(lamda)

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


def diffPrivate_model(ins, step_len,
                      epsilon, delta, init_mod,
                      f, heavy_ball, beta,
                      sensitivity_delta):
    import numpy as np
    import codes.model_dp as model_dp
    import codes.tools as tools
    max_iter = 50  # number of iterations that the differentially private algorithm is ran

# Upper Limit creation for each party
    # if np.size(upper_limit_coef) > 1:
    #     upper_limit = upper_limit_coef
    # elif np.size(upper_limit_coef) == 1:
    #     upper_limit = upper_limit_coef * np.ones(parties)
    #
    #
    # Obtaining upperbounds for step sizes
    optimal_lamdas = init_mod.Pi[-ins.m:]
    M_sq = np.linalg.norm(optimal_lamdas)**2
    sigma = sum([np.linalg.norm(np.ones(ins.m))**2 for i in range(ins.parties)])
    s_kappa_norm_sq = np.linalg.norm(ins.m*np.ones(ins.parties))**2
    B_sq = (2*max_iter*max_iter*sigma)/(epsilon*epsilon)+s_kappa_norm_sq
    #####
# Differentially private subgradient algorithm
    # lambda is initialized as 0 vector
    # while not coverged
    #   each party solves its subproblem, returns noisy s_k's
    #   (and also normal s_k) if better solution obtained (using noisy s_k's),
    #   then it is stored lambda values are updated
    lamda_dp = np.median(ins.r)*np.zeros(ins.m)
    iteration = 1
    lamda_dp_vals = []
    lamda_dp_vals.append(lamda_dp)
    best_obj_vals = []
    best_obj_val_dp = init_mod.objVal * 1000
    noisy_s_vals_t = [np.zeros(ins.m) for _ in range(ins.parties)]
    if f is not None:
        f.write('1st Mdl Obj\t|Prvt Mdl\tPrvt Min\t#Iter\n')
        f.write('--------------------------------------------------------------\n')
    rho = tools.rhoCalculator(epsilon, delta)
    rho_0 = rho / (max_iter * ins.m)

    alfa = 2
    C = alfa * ins.c
    S_k_0 = C  # / ins.parties
    S_values = np.array([S_k_0 for _ in range(ins.parties)])

    num_of_iterations = max_iter

    while iteration < num_of_iterations:
        # Solves subproblems and returns the summation
        #   of noisy and normal s_k's and each submodels
        (opt_mod_dp, s_total_noisy, s_total_trun, s_total_real, noisy_s_vals,
         x_val_dp,
         common_cap_duals_dp,
         private_cap_duals_dp) = model_dp.optDist(ins, rho_0,
                                                  lamda_dp,
                                                  noisy_s_vals_t,
                                                  sensitivity_delta,
                                                  S_values)

        # Update S_values according to the noisy_s_vals
        # for p in range(ins.parties):
        #     S_values[p] = np.divide(np.multiply(C, noisy_s_vals[p]), np.sum(noisy_s_vals, axis=0))
        noisy_s_vals_t = noisy_s_vals
        # Keeps the best solution until now
        current_obj_val_dp = (sum(opt_mod_dp[i].objVal for i in range(ins.parties))
                              + np.inner(ins.c, lamda_dp))

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
        if (f is not None) and (iteration % 10 == 0 or iteration == 1):
            f.write('%.5f\t|%.5f\t%.5f\t%d\n' %
                    (init_mod.objVal,
                     current_obj_val_dp,
                     best_obj_val_dp,
                     min_iter_dp
                     ))
        # Lambda updates
        gradient_dp = s_total_trun - ins.c
        if max(abs(gradient_dp)) <= 1.0e-6:
            break
        # if step_len == 1:
        #     nu = 1.0/(iteration**(0.5))
        # elif step_len == 2:
        #     nu = 1.0/(iteration**(0.51))

        # IF(MOD(iteration,10)=0,scaler/POWER(iteration,0.5),scaler)
        scaler = ins.parties**1.5
        # if iteration % 10 == 0:
        #     scaler = scaler / np.sqrt(iteration)

        if heavy_ball:
            if iteration == 1:
                pi = (beta * B_sq * max_iter) / (1-beta)
                gamma = beta * M_sq
                N = current_obj_val_dp - init_mod.objVal
                psi = (2 * beta * N) / (1-beta)
            nu = scaler * (psi*np.sqrt((pi**2+4*psi*gamma)/(psi**2))-pi)/(2*psi)
        else:
            nu = scaler * np.sqrt(M_sq)/(np.sqrt(B_sq*max_iter))

        # if step_len == 1:
        #     nu = 1.0/(iteration**(0.5))

        if heavy_ball and iteration > 2:
            lamda_dp = np.maximum(0, (lamda_dp +
                                      nu * gradient_dp +
                                      beta * (lamda_dp - lamda_dp_vals[-2])))
        else:
            lamda_dp = np.maximum(0, lamda_dp + nu * gradient_dp)

        # Increase the iteration number
        iteration = iteration + 1
        lamda_dp_vals.append(lamda_dp)
    # Real usage on the last iteration and on the best solution in the DP Algorithm
    ss_best_total = []
    for j in range(len(ins.c)):
        s = 0
        for i in range(ins.parties):
            s = s + s_total_real[i][j]
        ss_best_total.append(s)

    ss_best_total_real = []
    for j in range(len(ins.c)):
        s = 0
        for i in range(ins.parties):
            s = s+best_s_total_real[i][j]
        ss_best_total_real.append(s)
    if f is not None:
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

    return (best_obj_vals, best_obj_val_dp, best_x_val_dp, best_private_cap_duals_dp,
            best_common_cap_duals_dp, s_total_real)


def result_row(ins,
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

    returned = -2*np.ones(3+3*ins.parties*ins.max_product +
                          ins.m + ins.parties * ins.max_private_cap +
                          ins.parties * (ins.m + ins.max_private_cap) +
                          ins.parties * (ins.m + ins.max_private_cap))
    # recording objective function values
    returned[0] = init_mod.objVal   # initial problem objective function value
    returned[1] = best_obj_val            # distributed problem objective function value
    returned[2] = best_obj_val_dp         # differentially private problem objective function value

    # recording x_k variable values
    returned[3:(3+sum(ins.n))] = [i.X for i in init_mod.getVars()[:sum(ins.n)]]              # initial problem x_k values
    returned[(3 + ins.parties*ins.max_product):(3+ins.parties*ins.max_product+sum(ins.n))] = best_x_val  # distributed problem x_k values
    returned[(3 + 2*ins.parties*ins.max_product):(3+2*ins.parties*ins.max_product+sum(ins.n))] = best_x_val_dp  # differentially private problem x_k values

    # index update
    index = 3 + 3*ins.parties*ins.max_product

    # recording dual variable values of the initial model (first shared, then private constraints)
    duals_optimal = -1 * np.ones(ins.m + ins.parties * ins.max_private_cap)

    duals_optimal[:ins.m] = init_mod.Pi[-ins.m:]
    private_cap_dual_optimal = init_mod.Pi[(3*ins.m*ins.parties):(3*ins.m*ins.parties+sum(ins.m_parties))]

    for i in range(ins.parties):
        temp = np.array(private_cap_dual_optimal[sum(ins.m_parties[0:i]):sum(ins.m_parties[0:i+1])])
        if len(temp) < ins.max_private_cap:
            temp = np.append(temp, -1*np.ones(ins.max_private_cap-len(temp)))

        duals_optimal[(ins.m + i * ins.max_private_cap):(ins.m + (i + 1) * ins.max_private_cap)] = temp

    returned[index:(index + ins.m + ins.parties * ins.max_private_cap)] = duals_optimal

    # recording dual variable values of the distributed model
    # index update
    index = index + ins.m + ins.parties * ins.max_private_cap
    #   note that each party will have maximum of (m+10) dual variables
    for i in range(ins.parties):
        temp = np.concatenate((best_common_cap_duals[:, i],
                               best_private_cap_duals[sum(ins.m_parties[0:i]):sum(ins.m_parties[0:i+1]), i]))
        if len(temp) < ins.m+ins.max_private_cap:
            temp = np.concatenate((temp, -1*np.ones(ins.m+ins.max_private_cap-len(temp))))
        returned[(index + i * (ins.m + ins.max_private_cap)):(index + (i + 1) * (ins.m + ins.max_private_cap))] = temp

    # index update
    index = index + ins.parties * (ins.m + ins.max_private_cap)

    # recording dual variable values of the differentially private model
    for i in range(ins.parties):
        temp = np.concatenate((best_common_cap_duals_dp[:, i],
                               best_private_cap_duals_dp[sum(ins.m_parties[0:i]):sum(ins.m_parties[0:i+1]), i]))
        if len(temp) < ins.m+ins.max_private_cap:
            temp = np.concatenate((temp, -1*np.ones(ins.m+ins.max_private_cap-len(temp))))
        returned[(index + i * (ins.m + ins.max_private_cap)):(index + (i + 1) * (ins.m + ins.max_private_cap))] = temp

    return returned


def simulation(parameter_list):
    import numpy as np
    import csv
    from codes.tools import instance
    parties = int(parameter_list[0])
    m = int(parameter_list[1])
    step_len = int(parameter_list[2])
    seed = int(parameter_list[3])
    max_private_cap = 10
    max_product = 15

    name = ('parties_' + str(parties) +
            '_sharedCap_' + str(m) +
            '_seed_' + str(seed) + '.txt')
    # Create data
    ins = instance(parties, m, max_private_cap, max_product, seed)
    # n, m_parties, r, A, B, c, c_individual = data_creation(parties, m,
    #                                                        max_private_cap,
    #                                                        max_product,
    #                                                        seed)
    sensitivity_delta_param = 0.1

    # Solve initial model with and without demand
    original_model, original_model_demand, init_mod, ins.product_demand = init_model(ins)

    f = open('./results/momentum_'+name, "w+")
    f.write('Objective value of the model without demand:\t' + str(original_model.objVal) +
            '\nObjective value of the model with demand:   \t' + str(original_model_demand.objVal)
            + '\n')

    # Initial model where everybody input their information to the model, also s_k is introduced
    f.write('\t\tModel with additional constraints (s_k included model)\n')
    f.write('----------------------------------------------------------------------------------\n')
    f.write('Obj:\t' + str(init_mod.objVal) + '\n')

    # Solve data private model
    (obj_vals,
     best_obj_val,
     best_x_val,
     best_private_cap_duals,
     best_common_cap_duals) = dataPrivate_model(ins, step_len,
                                                init_mod,
                                                f, heavy_ball=True, beta=0.9)
    f.close()
    obj_vals.insert(0, seed)
    obj_vals.insert(1, parties)
    with open('./results/momentum_dataPrivate.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(obj_vals)

    f = open('./results/'+name, "w+")
    f.write('Objective value of the model without demand:\t' + str(original_model.objVal) +
            '\nObjective value of the model with demand:   \t' + str(original_model_demand.objVal)
            + '\n')

    # Initial model where everybody input their information to the model, also s_k is introduced
    f.write('\t\tModel with additional constraints (s_k included model)\n')
    f.write('----------------------------------------------------------------------------------\n')
    f.write('Obj:\t' + str(init_mod.objVal) + '\n')
    (obj_vals,
     best_obj_val,
     best_x_val,
     best_private_cap_duals,
     best_common_cap_duals) = dataPrivate_model(ins, step_len, init_mod, f, heavy_ball=False, beta=0.9)
    f.close()
    obj_vals.insert(0, seed)
    obj_vals.insert(1, parties)
    with open('./results/dataPrivate.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(obj_vals)
    # Solve differentially private model for all epsilon and delta values
    for epsilon in [10.00]:  # [0.05,0.10,0.15,0.20,0.25]:
        for delta in [0.0010, 0.0100, 0.0500, 0.1000, 0.1500, 0.2000]:
            name = ('parties_' + str(parties) +
                    '_sharedCap_' + str(m) +
                    '_seed_' + str(seed) +
                    '_eps_' + '%.2f' % epsilon +
                    '_delta_' + '%.4f' % delta + '.txt')
            f = open('./results/' + name, "w+")
            (obj_vals_dp,
             best_obj_val_dp,
             best_x_val_dp,
             best_private_cap_duals_dp,
             best_common_cap_duals_dp,
             s_total_real) = diffPrivate_model(ins, step_len,
                                               epsilon, delta, init_mod,
                                               f, heavy_ball=False, beta=0.9,
                                               sensitivity_delta=sensitivity_delta_param)
            f.close()
            obj_vals_dp.insert(0, seed)
            obj_vals_dp.insert(1, list(np.ones(parties)))
            obj_vals_dp.insert(2, ins.c)
            for k in range(10):
                if k < parties:
                    obj_vals_dp.insert(k+3, s_total_real[k])
                else:
                    obj_vals_dp.insert(k+3, ' ')
            obj_vals_dp.insert(13, epsilon)
            obj_vals_dp.insert(14, delta)
            with open('./results/diffPrivate.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(obj_vals_dp)
            returned = result_row(ins,
                                  init_mod,
                                  best_obj_val, best_x_val, best_obj_val_dp, best_x_val_dp,
                                  best_common_cap_duals, best_private_cap_duals,
                                  best_common_cap_duals_dp, best_private_cap_duals_dp)

            a = np.hstack(((parties, m, epsilon, delta,
                            list(np.concatenate((np.ones(parties),
                                                 np.zeros(20-parties)), axis=None)),
                            seed, returned)))

            with open('./results/results.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(a)

    # HEAVY DP
    # Solve differentially private model for all epsilon and delta values

    for epsilon in [10.00]:  # [0.05,0.10,0.15,0.20,0.25]:
        for delta in [0.0010, 0.0100, 0.0500, 0.1000, 0.1500, 0.2000]:
            name = ('parties_' + str(parties) +
                    '_sharedCap_' + str(m) +
                    '_seed_' + str(seed) +
                    '_eps_' + '%.2f' % epsilon +
                    '_delta_' + '%.4f' % delta + '.txt')
            f = open('./results/momentum_' + name, "w+")
            (obj_vals_dp,
             best_obj_val_dp,
             best_x_val_dp,
             best_private_cap_duals_dp,
             best_common_cap_duals_dp,
             s_total_real) = diffPrivate_model(ins, step_len,
                                               epsilon, delta, init_mod,
                                               f, heavy_ball=True, beta=0.9,
                                               sensitivity_delta=sensitivity_delta_param)
            f.close()
            obj_vals_dp.insert(0, seed)
            obj_vals_dp.insert(1, list(np.ones(parties)))
            obj_vals_dp.insert(2, ins.c)
            for k in range(10):
                if k < parties:
                    obj_vals_dp.insert(k+3, s_total_real[k])
                else:
                    obj_vals_dp.insert(k+3, ' ')
            obj_vals_dp.insert(13, epsilon)
            obj_vals_dp.insert(14, delta)
            with open('./results/momentum_diffPrivate.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(obj_vals_dp)
            returned = result_row(ins,
                                  init_mod,
                                  best_obj_val, best_x_val, best_obj_val_dp, best_x_val_dp,
                                  best_common_cap_duals, best_private_cap_duals,
                                  best_common_cap_duals_dp, best_private_cap_duals_dp)

            a = np.hstack(((parties, m, epsilon, delta,
                            list(np.concatenate((np.ones(parties),
                                                 np.zeros(20-parties)), axis=None)),
                            seed, returned)))

            with open('./results/momentum_results.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(a)

    return 1


if __name__ == "__main__":
    from codes.tools import cartesian
    grid = list(cartesian(([5, 8, 10, 20],   # number of parties
                           [5],             # number of shared capacities
                           [1],             # 1: step_len nu^(0.50), 2: constant
                           range(1, 31))))  # seed

    for i in range(len(grid)):
        simulation(grid[i])
        print('Parties: {} Seed: {}'.format(grid[i][0], grid[i][3]))
