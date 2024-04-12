import numpy as np

from input import Simulation_Parameters
from dumper import Dumper
from functions import Functions
from bulk_damage import BulkDamage
from solve import Solver
from postprocess import PostProcess

def main(parameters, incs) :

    N_elements = parameters.N_elements

    u = np.zeros(2 * N_elements)
    d_prev = np.zeros(N_elements - 1)
    d = np.zeros(N_elements - 1)
    lambda_ = np.zeros(N_elements - 1)

    bc = {0 : 0, (2*N_elements-1) : 0}

    functions = Functions (parameters)
    solver = Solver(functions, parameters)
    dumper = Dumper()

    strain_str = []
    stress_str = []
    jump_str = []


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
        functional = solver.functional.assemble_clip_functional(d, D, u_, lambda_, bc)
        cde = solver.functional.get_cohesive_dissipation(d)
        bde = solver.functional.get_bulk_dissipation(D)

        strain_str.append(solver.functional.get_strain(u_))
        stress_str.append(F_[-1])
        jump_str.append(solver.functional.get_jump(u_))

        cda,bda = solver.functional.dissipation_act_bulk_coh(strain_str,stress_str,jump_str)

        dumper.store("imposed_displacement", u_t)
        dumper.store("force", F_[-1])
        dumper.store("displacement", u_)
        dumper.store("cohesive_damage", d)
        dumper.store("bulk_damage", D)
        dumper.store("lagrange", lambda_)
        dumper.store("cohesive_stress", parameters.k * solver.functional.get_jump(u_) * functions.gd_cohesive(d))
        dumper.store("cohesive_dissip_expected",cde)
        dumper.store("bulk_dissip_expected", bde)
        dumper.store("total_dissip_expected",cde + bde)
        dumper.store("cohesive_dissip_actual",cda)
        dumper.store("bulk_dissip_actual",bda)
        dumper.store("total_dissip_actual",cda + bda)

    

    return dumper

if __name__ == '__main__':

    E = 3e10
    Gc = 120
    sigc = 3e6
    L = 0.2
    Dm = 0.7
    alpha = np.pi/6
    beta = 0
    he = 20
    choice = 1
    N_increments = 30
    max_iter = 100

    alpha_values = [0.5,0.4,0.3]
    Dm_values = [0.7,0.8,0.9]
    choice_values = [2]

    post_processes = []

    if len(alpha_values) == 1 and len(Dm_values) ==1 and len(choice_values) == 1:
        alpha = alpha_values[0]
        Dm = Dm_values[0]
        choice = choice_values[0]

        parameters = Simulation_Parameters(E, Gc, sigc, L, Dm, alpha, beta, he, choice ,N_increments, max_iter)
        parameters.alpha_values = alpha_values 
        parameters.Dm_values = Dm_values
        incs =  np.concatenate((np.array([0]), np.linspace(parameters.epsilon_0*L, parameters.wc*1.5, N_increments-1)))

        main_run = main(parameters, incs)
        exact_inc, exact_f = parameters.pure_czm()

        post_process = PostProcess(parameters, [main_run], exact_inc, exact_f)
        post_process.plot_stress_ut()
        post_process.plot_dissip_all()

    if len(choice_values)==1 and (len(alpha_values) != 1 or len(Dm_values) != 1):
        choice = choice_values[0]
        dumpers = []
        for alpha in alpha_values:
            for Dm in Dm_values:

                parameters = Simulation_Parameters(E, Gc, sigc, L, Dm, alpha, beta, he, choice ,N_increments, max_iter)
                parameters.alpha_values = alpha_values
                parameters.Dm_values = Dm_values
                incs =  np.concatenate((np.array([0]), np.linspace(parameters.epsilon_0*L, parameters.wc*1.5, N_increments-1)))

                main_run = main(parameters, incs)
                exact_inc, exact_f = parameters.pure_czm()
                dumpers.append(main_run)

        post_process = PostProcess(parameters, dumpers, exact_inc, exact_f)
        post_process.plot_stress_ut()
        post_process.plot_dissip_all()

    if len(choice_values)!=1 and len(alpha_values)!=1 and len(Dm_values)!=1 :
        dumpers = []
        comp_coh_disp =[]
        comp_bulk_disp =[]
        alpha_plot = []
        Dm_plot =[]


        for choice in choice_values:
            comp_coh_disp_temp =[]
            comp_bulk_disp_temp =[]
            alpha_plot_temp = []
            Dm_plot_temp = []
            for alpha in alpha_values:
                for Dm in Dm_values:
                    alpha_plot_temp.append(alpha)   
                    Dm_plot_temp.append(Dm)
                    parameters = Simulation_Parameters(E, Gc, sigc, L, Dm, alpha, beta, he, choice ,N_increments, max_iter)
                    parameters.alpha_values = alpha_values
                    parameters.Dm_values = Dm_values
                    parameters.choice_values = choice_values
                    incs =  np.concatenate((np.array([0]), np.linspace(parameters.epsilon_0*L, parameters.wc*1.5, N_increments-1)))

                    main_run = main(parameters, incs)
                    comp_coh_disp_temp.append(np.max(main_run.data["cohesive_dissip_actual"]))
                    comp_bulk_disp_temp.append(np.max(main_run.data["bulk_dissip_actual"]))
                    exact_inc, exact_f = parameters.pure_czm()
                    dumpers.append(main_run)
            
            comp_coh_disp.append(comp_coh_disp_temp)
            comp_bulk_disp.append(comp_bulk_disp_temp)
            alpha_plot.append(alpha_plot_temp)
            Dm_plot.append(Dm_plot_temp)
            
        # print("alpha_plot = ",alpha_plot)
        # print("Dm_plot = ",Dm_plot)
        # print("comp_coh_disp =",comp_coh_disp)
        # print("comp_bulk_disp =", comp_bulk_disp)
        # print("Dm_values = ",Dm_values)
        # print("choice_values = ",choice_values)
        # print("alpha_values = ",alpha_values)
        #post_process = PostProcess(parameters, dumpers, exact_inc, exact_f,comp_coh_disp,comp_bulk_disp,alpha_plot,Dm_plot,alpha_values,Dm_values,choice_values)
        post_process.plot_dissip_comp()
        
        



        

        
        



        
