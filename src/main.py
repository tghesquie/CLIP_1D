""" Main file"""
import os
import uuid
import numpy as np

from input import Simulation_Parameters
from functions import Functions_4_terms, Functions_3_terms, Functions_CZM, Functions_Lip
from solve import Solver

# pylint: disable = C0303
def initialize_parameters(damage_function = None, Dm = None, alpha = None, beta = None):
    return Simulation_Parameters(E, Gc, sigc, L, Dm, alpha, beta, he, functional_choice, damage_function, N_increments, max_iter)

def generate_filename(base="results"):
    """
    Genreates a Unique ID for each main run
    """
    unique_id = uuid.uuid4()
    return f"{base}_{unique_id}.npz"

def main_clip(parameters, incs, terms):
    """
    Main function for the CLIP functional with specified terms
    """
    print("------------------------------------------------")
    print(f"Start : {parameters.functional_choice}, Dm = {parameters.Dm}")
    print("------------------------------------------------")

    N_elements = parameters.N_elements
    num_incs = len(incs)

    u_ = np.zeros(2 * N_elements)
    d_prev = np.zeros(N_elements - 1)
    d = np.zeros(N_elements - 1)
    lambda_ = np.zeros(N_elements - 1)
    bc = {0: 0, (2*N_elements-1): 0}

    functions = Functions_4_terms(parameters) if terms == 4 else Functions_3_terms(parameters)
    solver = Solver(functions, parameters)

    strain_str = []
    stress_str = []
    jump_str = []
    displacement_str = []
    coh_damage_str = []
    bulk_damage_str = []
    lambda_str = []
    cde_str = np.zeros(num_incs)
    cda_str = np.zeros(num_incs)
    bde_str = np.zeros(num_incs)
    bda_str = np.zeros(num_incs)
    tda_str = np.zeros(num_incs)
    tde_str = np.zeros(num_incs)

    for i, u_t in enumerate(incs):
        bc[(2*N_elements-1)] = u_t
        if i == 2:
            d_prev = np.zeros(N_elements - 1)
            d_prev[int((N_elements - 1)/2)] = 0.01
        else:
            d_prev = d.copy()

        print("--------------------------------")
        print(f"Increment : {i}, ut = {u_t}")

        results = solver.solve_functional(d, d_prev, bc)
        d = results.x

        D = solver.bulk_damage.get_Bulk_damage(d) 
        u_, F_, lambda_, _ = solver.equilibrium_solver.solve_equilibrium_ul(d, D, bc)
        cde = solver.functional.get_cohesive_dissipation(d)
        bde = solver.functional.get_bulk_dissipation(D) if terms == 4 else None

        strain_str.append(solver.functional.get_strain(u_))
        stress_str.append(F_[-1])
        jump_str.append(solver.functional.get_jump(u_))
        displacement_str.append(u_)
        coh_damage_str.append(d)
        bulk_damage_str.append(D)
        lambda_str.append(lambda_)

        t1, t2, t3 = solver.functional.get_cohesive_energy_lagrange(solver.functional.get_jump(u_), lambda_, d)
        cda, bda = solver.functional.dissipation_act_bulk_coh(strain_str, stress_str, jump_str)
        bda = bda - solver.functional.get_strain_energy(solver.functional.get_strain(u_), D)
        cda = cda - (-t1 + t2 - t3)

        cde_str[i] = cde
        cda_str[i] = cda
        bda_str[i] = bda
        bde_str[i] = bde
        tda_str[i] = cda + bda
        tde_str[i] = cde

    results_dict = {
        'inputs': parameters.to_dict(),
        'imposed_disp': incs,
        'stress': stress_str,
        'seperation': jump_str,
        'displacement': displacement_str,
        'cohesive_damage': coh_damage_str,
        'bulk_damage': bulk_damage_str,
        'lmb': lambda_str,
        'strain': strain_str,
        'coh_disp_act': cda_str,
        'bulk_disp_act': bda_str,
        'tot_disp_act': tda_str,
        'coh_disp_exp': cde_str,
        'bulk_disp_exp': bde_str if terms == 4 else None,
        'tot_disp_exp': tde_str,
    }
    filename = generate_filename()
    np.savez(filename, **results_dict)
    print("------------------------------------------------")
    print(f"End : {parameters.functional_choice}, Dm = {parameters.Dm}")
    print("------------------------------------------------")

    return results_dict

def main_clip_4_terms (parameters, incs):
    """
    Main function for CLIP functional with 4 terms
    """
    result = main_clip(parameters, incs, terms = 4)
    return result

def main_clip_3_terms (parameters, incs):
    """
    Main function for CLIP functional with 3 terms
    """
    result = main_clip(parameters, incs, terms = 3)
    return result

def main_czm(parameters, incs):
    """
    Main for the CZM model 
    """
    print("------------------------------------------------")
    print("Start :", parameters.functional_choice)
    print("------------------------------------------------")

    N_elements = parameters.N_elements
    u = np.zeros(2 * N_elements)
    d_prev = np.zeros(N_elements - 1)
    d = np.zeros(N_elements - 1)
    lambda_ = np.zeros(N_elements - 1)

    bc = {0 : 0, (2*N_elements-1) : 0}

    functions = Functions_CZM(parameters)
    solver = Solver(functions, parameters)

    displacement_str = []
    coh_damage_str = []    
    lambda_str = []
    stress_str = []
    strain_str =[]
    jump_str =[]

    for i, u_t in enumerate(incs):
        bc[(2*N_elements-1)] = u_t 
        d_prev = d.copy()

        print("--------------------------------")
        print("Increment : ",i , "ut =",u_t)

        res = solver.solve_functional(d,d_prev ,bc)
        d = res.x
        D_pseudo_czm = (np.concatenate([np.ones(1), np.ones_like(d)]))
        u_, F_, lambda_, _ = solver.equilibrium_solver.solve_equilibrium_ul(d,D_pseudo_czm,bc)

        displacement_str.append(u_)
        coh_damage_str.append(d)
        lambda_str.append(lambda_)

        stress_str.append(F_[-1])
        strain_str.append(solver.functional.get_strain(u_))
        jump_str.append(solver.functional.get_jump(u_))

    results_dict = {
        'inputs': parameters.to_dict(),
        'imposed_disp': incs,
        'stress': stress_str,
        'seperation': jump_str,
        'displacement': displacement_str,
        'cohesive_damage': coh_damage_str,
        'lmb': lambda_str,
    }
    filename = generate_filename()
    np.savez(filename, **results_dict)

    print("------------------------------------------------")
    print("End :", parameters.functional_choice)
    print("------------------------------------------------")

    return results_dict

def main_lip(parameters, incs):
    """
    Main for the LIP model 
    """
    print("------------------------------------------------")
    print("Start :", parameters.functional_choice)
    print("------------------------------------------------")

    N_elements = parameters.N_elements
    N_v = parameters.N_v
    max_iter = parameters.max_iter
    u0 = np.zeros(N_v)
    D0 = np.zeros(N_elements)
    u = u0.copy()
    D = D0.copy()
    iteration = 0
    stop = False
    tol = 1e-6

    bc = {0 : 0, (N_v-1) : 0}

    functions = Functions_Lip(parameters)
    solver = Solver(functions, parameters)

    displacement_str = []    
    bulk_damage_str = []
    stress_str = []
    strain_str =[]   

    for i, u_t in enumerate(incs):

        print('inc', i, 'imposed u(L) : ',  u_t)
        bc[N_v-1] = u_t
        iteration = 0
        low_bound = D.copy()
        stop = False

        print("--------------------------------")
        print("Increment :",i , "ut =",u_t)

        while (iteration < max_iter) and not stop:
            u0 = u.copy()

            if i == 2 and iteration == 0:
                D0 = np.zeros(N_v-1)
                D0[int((N_v -1)/2)] = 0.1
            else :
                D0 = D.copy()
            
            u_fun,F_fun  = solver.equilibrium_solver_lip.solve_equilibrium_ul(D0 , bc)      
                  
            res = solver.lip_functional(D0,u_fun, low_bound)  
            D = res.x

            iteration += 1
            normdeltad = parameters.dx*np.linalg.norm(D-D0)
            normdeltau = parameters.dx*np.linalg.norm(u-u0)
            
            print("Iteration :",iteration)
            print("Norm_delta_d = ", normdeltad,"Norm_delta_u = ",normdeltau)
            if ( normdeltad < tol) & (normdeltau <= tol) : 
                stop = True
            
        displacement_str.append(u_fun)
        stress_str.append(F_fun[-1])
        bulk_damage_str.append(D)
        strain_str.append(solver.functional.get_strain_lip(u_fun))
            
    results_dict = {
        'inputs': parameters.to_dict(),
        'imposed_disp': incs,
        'stress': stress_str,
        'displacement': displacement_str,
        'bulk_damage': bulk_damage_str,
        'strain': strain_str,
    }
    filename = generate_filename()
    np.savez(filename, **results_dict )
    
    print("------------------------------------------------")
    print("End :", parameters.functional_choice)
    print("------------------------------------------------")

    return results_dict

def main_exact_pure_czm(parameters):
    """
    Main function to calculate the exact (Pure CZM) solution
    """

    print("------------------------------------------------")
    print("Start :", parameters.functional_choice)
    print("------------------------------------------------")


    exact_f = [0, parameters.sigc, 0]
    exact_in = [0, (parameters.sigc * parameters.L)/parameters.E, parameters.wc]

    results_dict = {
        'inputs': parameters.to_dict(),
        'imposed_disp': exact_in,
        'stress': exact_f,
    }
    filename = generate_filename()
    np.savez(filename, **results_dict )
    
    print("------------------------------------------------")
    print("End :", parameters.functional_choice)
    print("------------------------------------------------")

    return results_dict

if __name__ == '__main__':

    ################################################################
    # Creates folder to store the results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    folder_name = generate_filename()
    results_folder = os.path.join(script_dir,folder_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)    
    os.chdir(results_folder)

    ################################################################
    # For functional_choice = 'CLIP-4terms' : (To run the CLIP model)
    # damage function choices :
    #        - damage_function = cos_sin (require parameter : alpha and beta)
    #       - damage_function = D_squared (require parameter : alpha and beta)
    #       - damage_function = D_std (require parameter : alpha and beta)

    # For functional_choice = 'CLIP-3terms' : (To run the CLIP model with Single dissipation term)
    #  damage function choices :
    #       - damage_function = cos_sin_D_squared (require parameter : alpha)

    # For functional_choice = 'CZM': (To run the CZM model)
    #    damage function choices :
    #        - damage_function = CZM

    # For functional_choice = 'LIP' : (To run the LIP model)
    #    damage function choices :
    #       - damage_function = LIP (require parameter : alpha)

    # For functional_choice = 'Exact' :(To obatin the exact solution of Pure CZM problem)
    ####################################################################

    E = 3e10
    Gc = 120
    sigc = 3e6
    L = 0.2
    Dm = 0.9
    alpha = np.pi/4
    beta = 0
    he = 10
    functional_choice = 'CLIP-3terms'
    N_increments = 30
    max_iter = 100
    damage_function = 'cos_sin_D_squared'
    
    parameters_clip = initialize_parameters(damage_function, Dm, alpha,)
    incs_clip =  np.concatenate((np.array([0]), np.linspace(parameters_clip.epsilon_0*L, parameters_clip.wc*1.5, N_increments-1)))    
    result_1 = main_clip_3_terms(parameters_clip, incs_clip)
  
    ################################################################
    damage_function = 'CZM'
    functional_choice = 'CZM'
    parameters_czm = initialize_parameters(damage_function = damage_function)
    incs_czm =  np.concatenate((np.array([0]), np.linspace(parameters_czm.epsilon_0*L, parameters_czm.wc*1.5, N_increments-1))) 
    result_2 = main_czm(parameters_czm, incs_czm)

    ###############################################################
    damage_function = 'LIP'
    functional_choice = 'LIP'
    parameters_lip = initialize_parameters(damage_function = damage_function, alpha = alpha)
    incs_lip =  np.concatenate((np.array([0]), np.linspace(parameters_lip.epsilon_0*L, parameters_lip.wc*1.5, N_increments-1))) 
    result_3 = main_lip(parameters_lip, incs_lip)

    ############################################################
    functional_choice = 'Exact'
    parameters_exact = initialize_parameters()
    result_4 = main_exact_pure_czm(parameters_exact)

    ############################################################

    


        

        
        



        
