import sys
sys.path.insert(0,"/home/ssshetty/Home/Main/Akantu/akantu/build/python")
import akantu as aka
import numpy as np
import uuid
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from input import Explicit_Parameter, Clip_Explicit_Parameters
from functions import  Functions_explicit_czm,Functions_explicit_clip
from bulk_damage import BulkDamage
from solve import ExplicitSolver, Solver
from post_process import damage_comp_plot, energy_comp_plot, bulk_damage_at_nodes, strain_at_timestep, opening_along_time, GD_function_value_along_bar, velocity_along_the_bar, velocity_along_the_bar_3D, velocity_along_bar_at_timestep, displacement_along_bar_at_timestep, energies_comparison, acceleration_along_bar_at_timestep,energies_comparison_string,Stress_along_bar_at_time_step
from tqdm import tqdm
import logging

def initialize_parameters():
    return Explicit_Parameter(E, Gc, sigc, rho, L, Area, eps0dot, max_steps, N_elements, new_crack)

def initialize_parameters_clip():
    return Clip_Explicit_Parameters(E, Gc, sigc, rho, Area, eps0dot, L, Dm, max_steps, N_elements,new_crack,boundary_type, nlc)

def generate_filename(base="results_exp"):
    """
    Genreates a Unique ID for each main run
    """
    unique_id = uuid.uuid4()
    return f"{base}_{unique_id}.npz"

def Clip_no_opt_algorithm(dt, parameters):

    logging.basicConfig(level=logging.INFO)
    #logging.basicConfig(level=logging.DEBUG)

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
    jump = 0
    disp = np.zeros(N_nodes)
    vel = np.zeros(N_nodes+1)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    lda = np.zeros(N_elements-1)

    d_str = [] 
    stress = []
    Ep_str = [] 
    Edis_str = [] 
    Ekin_str = []
    Ecoh_str = []
    Ext_work_str = []
    Tot_str = []
    D_nodes_str = []
    GD_nodes_str = []

    disp_str = []
    vel_str = []
    
    acc_str = []
    nodes_str = []
    jump_str = []
    strain_str = []
    
    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'displacement': [],
        'displacement_crack': [],
        'velocity': [],
        'velocity_crack': [],
        'acceleration': [],
        'acceleration_crack': [],
        'nodes_str': [],
        'nodes_str_crack': [],
        'strain_str': [],
        'stress_str': [],
        'stress_crack_str': [],
        'damage_mean': [],
        'opening': [],
        'bulk_damage_nodes': [],
        'GD_function': [],
        'potential_energy': [],
        'kinetic_energy': [],
        'dissipated_energy': [],
        'dissipated_bulk_energy': [],
        'dissipated_coh_energy': [],
        'cohesive_energy': [],
        'external_work': [],
        'total_energy': [],
        'coh_dissipation_actual': [],
        'bulk_dissipation_actual' : [],
        'total_dissipation_actual' :  []
    }

    bulk_damage = BulkDamage(parameters.lc, parameters.Dm, parameters.get_len_mat(parameters.x))
    functions = Functions_explicit_clip(parameters)
    solver = ExplicitSolver(functions, parameters)
    
    disp, acc, vel_predict, d = solver.initial_and_boundary_conditions(nodes)
    
    cohesive_dissipation = 0.0
    traction_at_center = 0.0
    bulk_dissipation = 0.0
    previous_strain_energy = 0.0
    cohesive_dissipation_act = 0.0
    bulk_dissipation_act = 0.0
    prev_jump = 0.0
    previous_strain = 0.0
    prev_traction = 0.0
    previous_stress = 0.0
    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        
        time += dt
        logging.info(" Time %f", time)
        logging.info(" Time step %d", step)
        
        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp,vel ,acc, vel_predict)
        logging.debug(f"Displacement: {disp}")

        if parameters.new_crack != 0:
            N_nodes_half = math.ceil(parameters.N_nodes / 2) 
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            traction_at_center, d[math.floor((parameters.N_elements-1)/2)], jump = solver.traction_predict(jump, opt = False)
            trac_average = (traction_at_center + prev_traction )/2
            cohesive_dissipation += trac_average * (jump - prev_jump) * parameters.Area
            prev_jump = jump
            prev_traction = traction_at_center
        D_nodes, D, _ = bulk_damage.get_Bulk_damage(d, centeronly=False)
        logging.info(f"Damage at step {step}: {d}")

        stress, strain, stress_bulk = solver.get_stress(disp, d, lda)
        logging.debug(f"Stress: {stress}")

        force = solver.get_nodal_forces(stress)
        logging.debug(f"force: {force}")
        
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        logging.debug(f"Acceleration: {acc}")

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        logging.debug(f"Velocity: {vel}")
     
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)
   
        Ep, Ekin, Edis, Ext_work, Ecoh, Total_energy, Edissip_bulk, Edissip_coh  = solver.Energy_computation(disp, vel, lda, d, stress, dt,Ext_work)
        
        strain_increment = strain - previous_strain
        stress_average = (stress_bulk + previous_stress )/2
        bulk_dissipation_increment = parameters.dx * parameters.Area * np.dot(stress_average , strain_increment)
        bulk_dissipation += (bulk_dissipation_increment)
        bulk_dissipation_act = (bulk_dissipation - Ep)

        previous_strain = strain
        previous_stress = stress_bulk

        cohesive_dissipation_act = (cohesive_dissipation - Ecoh)
        total_dissipation = bulk_dissipation_act + cohesive_dissipation_act

        Total_energy = Ep + Ekin  + total_dissipation + Ecoh


        if parameters.new_crack != 0 :
            results_dict['nodes_str_crack'] = (solver.get_nodes())
            results_dict['displacement_crack'].append(np.copy(disp))
            results_dict['velocity_crack'].append(np.copy(vel))
            results_dict['acceleration_crack'].append(np.copy(acc))
            results_dict['stress_crack_str'].append(np.copy(stress))
            
        else :
            results_dict['nodes_str'] = (solver.get_nodes())
            results_dict['displacement'].append(np.copy(disp))
            results_dict['velocity'].append(np.copy(vel))
            results_dict['acceleration'].append(np.copy(acc))
            results_dict['stress_str'].append(np.copy(stress))
            
        results_dict['opening'].append(np.copy(jump))
        results_dict['damage_mean'].append(np.copy(d[math.floor((parameters.N_elements-1)/2)]))
        results_dict['bulk_damage_nodes'].append(np.copy(D_nodes))
        results_dict['GD_function'].append(np.copy(functions.GD_bulk.get_value(D_nodes)))
        results_dict['strain_str'].append(np.copy(strain))
        results_dict['dissipated_energy'].append(np.copy(Edis))
        results_dict['dissipated_bulk_energy'].append(np.copy(Edissip_bulk))
        results_dict['dissipated_coh_energy'].append(np.copy(Edissip_coh))
        results_dict['potential_energy'].append(np.copy(Ep))
        results_dict['kinetic_energy'].append(np.copy(Ekin))
        results_dict['cohesive_energy'].append(np.copy(Ecoh))
        results_dict['external_work'].append(np.copy(Ext_work))
        results_dict['total_energy'].append(np.copy(Total_energy))

        results_dict['coh_dissipation_actual'].append(np.copy(cohesive_dissipation_act))
        results_dict['bulk_dissipation_actual'].append(np.copy(bulk_dissipation_act))
        results_dict['total_dissipation_actual'].append(np.copy(total_dissipation))

    filename = generate_filename()
    np.savez(filename, **results_dict)

    return results_dict

if __name__ == "__main__" :

    L = 1
    Area = 0.1
    E = 3e9
    rho = 275
    sigc = 3e6
    Gc = 120
    eps0dot = 5
    max_steps = 300
    Dm = 0.0
    N_elements = 40
    new_crack = 0
    boundary_type = "string"
    nlc = 4
    
    ################################
    paramters_opti_clip = initialize_parameters_clip()
    
    c = np.sqrt(E/rho)
    dt = 0.1 * ((L/N_elements)/c)
    
    results_no_opti_clip = Clip_no_opt_algorithm(dt, paramters_opti_clip)
    
    Dm = 0.3
    paramters_opti_clip_1 = initialize_parameters_clip()
    results_no_opti_clip_1 = Clip_no_opt_algorithm(dt, paramters_opti_clip_1)
   
    Dm = 0.8
    paramters_opti_clip_2 = initialize_parameters_clip()
    results_no_opti_clip_2 = Clip_no_opt_algorithm(dt, paramters_opti_clip_2)

    ###############################
    ###Post Process###
    time_index = 20
    #strain_at_timestep(dt, time_index, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2,)
    #Stress_along_bar_at_time_step(dt, time_index,results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2 )
    energies_comparison_string(dt,results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2)