import numpy as np
import os
import uuid

from input import Simulation_Parameters
from functions import Functions_4_terms, Functions_3_terms,Functions_CZM,Functions_Lip
from bulk_damage import BulkDamage
from solve import Solver


def initialize_parameters(Dm = None,alpha = None,beta = None):
    return Simulation_Parameters(E, Gc, sigc, L, Dm, alpha, beta, he, functional_choice, damage_function, N_increments, max_iter)

def generate_filename(base="results"):
    """
    Genreates a Unique ID for each main run
    """
    unique_id = uuid.uuid4()
    return f"{base}_{unique_id}.npz"

def main_clip_4_terms(parameters, incs) :
    """
    Main for the CLIP functional with 4 terms
    """
    exact_inc, exact_f = parameters.pure_czm()
    N_elements = parameters.N_elements

    u = np.zeros(2 * N_elements)
    d_prev = np.zeros(N_elements - 1)
    d = np.zeros(N_elements - 1)
    lambda_ = np.zeros(N_elements - 1)

    bc = {0 : 0, (2*N_elements-1) : 0}

    functions = Functions_4_terms(parameters)
    solver = Solver(functions, parameters)
    
    strain_str = []
    stress_str = []
    jump_str = []

    displacement_str = []
    coh_damage_str = []
    bulk_damage_str = []
    lambda_str = []

    cde_str = []
    cda_str = []
    bde_str = []
    bda_str = []
    tda_str = []
    tde_str = []

    for i, u_t in enumerate(incs):

        bc[(2*N_elements-1)] = u_t 

        if i == 2 :
            d_prev = np.zeros(N_elements - 1)
            d_prev[int((N_elements - 1)/2)] = 0.01
        
        else :
            d_prev = d.copy()

        print("--------------------------------", i)

        results = solver.solve_functional(d,d_prev,bc)
        d  = results.x

        # Storing the results
        D = solver.bulk_damage.get_Bulk_damage(d_prev)
        u_, F_, lambda_, d_ = solver.equilibrium_solver.solve_equilibrium_ul(d, D, bc)
        functional = solver.functional.assemble_clip_functional_4_terms(d, D, u_, lambda_, bc)
        cde = solver.functional.get_cohesive_dissipation(d)
        bde = solver.functional.get_bulk_dissipation(D)

        strain_str.append(solver.functional.get_strain(u_))
        stress_str.append(F_[-1])
        jump_str.append(solver.functional.get_jump(u_))

        displacement_str.append(u_)
        coh_damage_str.append(d)
        bulk_damage_str.append(D)
        lambda_str.append(lambda_)

        cda,bda = solver.functional.dissipation_act_bulk_coh(strain_str,stress_str,jump_str)

        cde_str.append(cde)
        cda_str.append(cda)
        bde_str.append(bde)
        bda_str.append(bda)
        tda_str.append(cda + bda)
        tde_str.append(cde + bde)

    filename = generate_filename()
    np.savez(filename, imposed_disp = incs, stress = stress_str, seperation = jump_str,
            displacement = displacement_str,cohesive_damage = coh_damage_str, bulk_damage = bulk_damage_str,
            lmb = lambda_str, strain = strain_str,
            coh_disp_act = cda_str, bulk_disp_act = bda_str, tot_disp_act = tda_str, 
            coh_disp_exp = cde_str, bulk_disp_exp = bde_str, tot_disp_exp = tde_str,
            exact_stress = exact_f, exact_seperation = exact_inc)

def main_clip_3_terms(parameters, incs) :
    """
    Main for the CLIP functional with 3 terms
    """
  
    exact_inc, exact_f = parameters.pure_czm()
    N_elements = parameters.N_elements

    u = np.zeros(2 * N_elements)
    d_prev = np.zeros(N_elements - 1)
    d = np.zeros(N_elements - 1)
    lambda_ = np.zeros(N_elements - 1)

    bc = {0 : 0, (2*N_elements-1) : 0}

    functions = Functions_3_terms(parameters)
    solver = Solver(functions, parameters)
    
    strain_str = []
    stress_str = []
    jump_str = []
    jump_center_str = []

    displacement_str = []
    coh_damage_str = []
    bulk_damage_str = []
    lambda_str = []

    cde_str = []
    cda_str = []
    bde_str = []
    bda_str = []
    tda_str = []
    tde_str = []

    for i, u_t in enumerate(incs):

        bc[(2*N_elements-1)] = u_t 

        if i == 2 :
            d_prev = np.zeros(N_elements - 1)
            d_prev[int((N_elements - 1)/2)] = 0.01
        
        else :
            d_prev = d.copy()

        print("--------------------------------", i)

        results = solver.solve_functional(d,d_prev,bc)
        d  = results.x

        # Storing the results
        D = solver.bulk_damage.get_Bulk_damage(d)
        u_, F_, lambda_, d_ = solver.equilibrium_solver.solve_equilibrium_ul(d, D, bc)
        functional = solver.functional.assemble_clip_functional_3_terms(d, D, u_, lambda_, bc)
        cde = solver.functional.get_cohesive_dissipation(d)
        
        strain_str.append(solver.functional.get_strain(u_))
        stress_str.append(F_[-1])
        jump_str.append(solver.functional.get_jump(u_))
       

        displacement_str.append(u_)
        coh_damage_str.append(d)
        bulk_damage_str.append(D)
        lambda_str.append(lambda_)
        t1,t2,t3 = solver.functional.get_cohesive_energy_lagrange(solver.functional.get_jump(u_),lambda_,d)

        cda,bda = solver.functional.dissipation_act_bulk_coh(strain_str,stress_str,jump_str)
        bda = bda - solver.functional.get_strain_energy(solver.functional.get_strain(u_),D)
        cda = cda - (-t1+t2-t3)
       
        cde_str.append(cde)
        cda_str.append(cda)

        bda_str.append(bda)
        tda_str.append(cda + bda)
        tde_str.append(cde)
        
    
    filename = generate_filename()
    parameters_dict = parameters.to_dict()

    np.savez(filename,inputs= parameters_dict, imposed_disp = incs, stress = stress_str, seperation = jump_str,
            displacement = displacement_str,cohesive_damage = coh_damage_str, bulk_damage = bulk_damage_str,
            lmb = lambda_str, strain = strain_str,
            coh_disp_act = cda_str, bulk_disp_act = bda_str, tot_disp_act = tda_str, 
            coh_disp_exp = cde_str, bulk_disp_exp = bde_str, tot_disp_exp = tde_str,
            exact_stress = exact_f, exact_seperation = exact_inc)

def main_czm(parameters, incs):
    """
    Main for the CZM functional 
    """
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
    bulk_damage_str = []
    lambda_str = []
    stress_str = []
    strain_str =[]
    jump_str =[]
    cde_str = []
    cda_str = []
    bde_str = []
    bda_str = []
    tda_str = []
    tde_str = []

    for i, u_t in enumerate(incs):
        bc[(2*N_elements-1)] = u_t 
        d_prev = d.copy()

        print(" ------------------ ", i)

        res = solver.solve_functional(d,d_prev ,bc)
        d = res.x
        D_pseudo_czm = (np.concatenate([np.ones(1), np.ones_like(d)]))
        u_, F_, lambda_, d_ = solver.equilibrium_solver.solve_equilibrium_ul(d,D_pseudo_czm,bc)

        displacement_str.append(u_)
        coh_damage_str.append(d)
        lambda_str.append(lambda_)

        stress_str.append(F_[-1])
        strain_str.append(solver.functional.get_strain(u_))
        jump_str.append(solver.functional.get_jump(u_))

    filename = generate_filename()
    parameters_dict = parameters.to_dict()

    np.savez(filename,inputs= parameters_dict, imposed_disp = incs, stress = stress_str, seperation = jump_str,
            displacement = displacement_str,cohesive_damage = coh_damage_str, bulk_damage = bulk_damage_str,
            lmb = lambda_str, strain = strain_str,
            coh_disp_act = cda_str, bulk_disp_act = bda_str, tot_disp_act = tda_str, 
            coh_disp_exp = cde_str, bulk_disp_exp = bde_str, tot_disp_exp = tde_str,
            )

def main_lip(parameters, incs):
    """
    Main for the LIP functional 
    """

    N_elements = parameters.N_elements
    N_v = parameters.N_v
    max_iter = parameters.max_iter

    u0 = np.zeros(N_v)
    D0 = np.zeros(N_elements)
    u = u0.copy()
    D = D0.copy()
    iteration = 0
    stop = False
    tol = 1e-5

    bc = {0 : 0, (N_v-1) : 0}

    functions = Functions_Lip(parameters)
    solver = Solver(functions, parameters)

    displacement_str = []
    coh_damage_str = []
    bulk_damage_str = []
    lambda_str = []
    stress_str = []
    strain_str =[]
    jump_str =[]
    cde_str = []
    cda_str = []
    bde_str = []
    bda_str = []
    tda_str = []
    tde_str = []

    for i, u_t in enumerate(incs):

        print('inc', i, 'imposed u(L) : ',  u_t)
        bc[N_v-1] = u_t
        iteration = 0
        low_bound = D.copy()
        stop = False

        while (iteration < max_iter) and not stop:
            u0 = u.copy()

            if i == 2 and iteration == 0:
                D0 = np.zeros(N_v-1)
                D0[int((N_v -1)/2)] = 0.1
            else :
                D0 = D.copy()
            
            print(" ------------------ ", i)

            u_fun,F_fun  = solver.equilibrium_solver_lip.solve_equilibrium_ul(D0 , bc)      
                  
            res = solver.lip_functional(D0,u_fun, low_bound)  
            D = res.x

            iteration += 1
            normdeltad = parameters.dx*np.linalg.norm(D-D0)
            normdeltau = parameters.dx*np.linalg.norm(u-u0)
            #norm_delta_fun = np.linalg.norm(solver.functional.assemble_lip_functional(d, u_fun) - solver.functional.assemble_lip_functional(d0 , u0 ))/np.linalg.norm(solver.functional.assemble_lip_functional(d0,u0))
            print("it = ",iteration,"nrm_fun = ",normdeltad,normdeltau )
            if ( normdeltad< 1.e-6) & (normdeltau <=1.e-6) : stop = True
            # if norm_delta_fun <= tol :
            #   stop = True
            # print("it = ",iteration,"nrm_fun = ",norm_delta_fun )

        displacement_str.append(u_fun)
        stress_str.append(F_fun[-1])
        bulk_damage_str.append(D)
            

    filename = generate_filename()
    parameters_dict = parameters.to_dict()

    np.savez(filename,inputs= parameters_dict, imposed_disp = incs, stress = stress_str, seperation = jump_str,
            displacement = displacement_str,cohesive_damage = coh_damage_str, bulk_damage = bulk_damage_str,
            lmb = lambda_str, strain = strain_str,
            coh_disp_act = cda_str, bulk_disp_act = bda_str, tot_disp_act = tda_str, 
            coh_disp_exp = cde_str, bulk_disp_exp = bde_str, tot_disp_exp = tde_str,
            )


if __name__ == '__main__':
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    folder_name = generate_filename()
    results_folder = os.path.join(script_dir,folder_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)    
    os.chdir(results_folder)

    ################################################################

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
   
    
    parameters_clip = initialize_parameters(Dm, alpha, beta)
    incs_clip =  np.concatenate((np.array([0]), np.linspace(parameters_clip.epsilon_0*L, parameters_clip.wc*1.5, N_increments-1)))    
    main_run = main_clip_3_terms(parameters_clip, incs_clip)
  
    damage_function = 'CZM'
    functional_choice = 'CZM'
    parameters_czm = initialize_parameters()
    incs_czm =  np.concatenate((np.array([0]), np.linspace(parameters_clip.epsilon_0*L, parameters_clip.wc*1.5, N_increments-1))) 
    main_run = main_czm(parameters_czm, incs_czm)

    damage_function = 'LIP'
    functional_choice = 'LIP'
    parameters_lip = initialize_parameters(alpha=alpha)
    incs_lip =  np.concatenate((np.array([0]), np.linspace(parameters_clip.epsilon_0*L, parameters_clip.wc*1.5, N_increments-1))) 
    main_run = main_lip(parameters_lip, incs_lip)

    


        

        
        



        
