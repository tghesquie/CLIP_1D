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
from functions import  Functions_explicit_czm,Functions_explicit_clip, Functions_Lip
from bulk_damage import BulkDamage
from solve import ExplicitSolver, Solver
from post_process import damage_comp_plot, energy_comp_plot, bulk_damage_at_nodes, strain_at_timestep, opening_along_time, GD_function_value_along_bar, velocity_along_the_bar, velocity_along_the_bar_3D, velocity_along_bar_at_timestep, displacement_along_bar_at_timestep, energies_comparison, acceleration_along_bar_at_timestep,energies_comparison_string,Stress_along_bar_at_time_step
from tqdm import tqdm
import logging

def initialize_parameters():
    return Explicit_Parameter(E, Gc, sigc, rho, L, Area, eps0dot, max_steps, N_elements, new_crack)

def initialize_parameters_clip():
    return Clip_Explicit_Parameters(E, Gc, sigc, rho, Area, eps0dot, L, Dm, max_steps, N_elements,new_crack,boundary_type, nlc, damage_function)

def generate_filename(base="results_exp"):
    """
    Genreates a Unique ID for each main run
    """
    unique_id = uuid.uuid4()
    return f"{base}_{unique_id}.npz"

def Akantu_algorithm(dt,material_file, mesh_file, parameters ):

    aka.parseInput(material_file)
    spatial_dimension = 2
    mesh = aka.Mesh(spatial_dimension)
    mesh.read(mesh_file)

    nodes = mesh.getNodes()

    model = aka.SolidMechanicsModelCohesive(mesh)
    model.getElementInserter().setLimit(aka._x, 0.49, 0.51)
    model.initFull(_analysis_method = aka._explicit_lumped_mass, _is_extrinsic = True)

    model.initNewSolver(aka._explicit_lumped_mass)
    model.updateAutomaticInsertion()

    # Initialization for bulk vizualisation
    model.setBaseName('Bar')
    model.addDumpFieldVector('displacement')
    model.addDumpFieldVector('velocity')
    model.addDumpFieldVector('external_force')
    model.addDumpField('strain')
    model.addDumpField('stress')
    model.addDumpField('blocked_dofs')

    # Initialization of vizualisation for Cohesive model
    model.setBaseNameToDumper('cohesive elements', 'cohesive')
    model.addDumpFieldVectorToDumper('cohesive elements', 'displacement')
    model.addDumpFieldToDumper('cohesive elements', 'damage')
    model.addDumpFieldVectorToDumper('cohesive elements', 'tractions')
    model.addDumpFieldVectorToDumper('cohesive elements', 'opening')

    model.applyBC(aka.FixedValue(0., aka._x), 'left')
    #model.applyBC(aka.FixedValue(0., aka._y), 'bottom')
    #model.applyBC(aka.FixedValue(0., aka._y), 'top')
    #functor_r = FixedDisplacement(aka._x, L*eps0dot)

    model.getExternalForce()[:] = 0

    disp_field = np.zeros(nodes.shape)
    disp_field[:, 0] = (nodes[:, 0])* (parameters.sigc * parameters.L)/(parameters.E)
    model.getDisplacement()[:] = disp_field

    model.applyBC(aka.FixedValue(0., aka._x), 'right')
    model.solveStep('explicit_lumped')
    model.checkCohesiveStress()
    
    time = 0
    model.dump()
    #dt = model.getStableTimeStep()*0.1
    model.setTimeStep(dt)

    damage_mean = []
    E_pot = []
    E_kin = []
    E_dis = []
    E_rev = []
    E_con = []
    Ext_str = []
    ext_test = 0
    tot_str = []

    for i in tqdm(range(0, parameters.max_steps)):
        time = time + dt
        print("i = ", i )
        print("time = ", time)

        # functor_r.set_time(time)
        # model.applyBC(functor_r, 'right')

        model.dump()
        model.dump('cohesive elements')
        
        model.checkCohesiveStress()
        model.solveStep('explicit_lumped')
     
        Evar_pot = model.getEnergy("potential")
        Evar_kin = model.getEnergy("kinetic")
        Evar_dis = model.getEnergy("dissipated")
        Evar_coh = model.getEnergy("reversible")
        
        stress_test = ( model.getMaterial(0).getStress(aka._triangle_3))
        vel_test = model.getVelocity()
  
        ext_test += -((stress_test[0][0]*  vel_test[0][0] * parameters.Area * dt) + (stress_test[2][0]* vel_test[2][0] *parameters.Area * dt))

        tot = Evar_pot + Evar_kin + Evar_dis + Evar_coh
        
        E_pot.append(Evar_pot)
        E_kin.append(Evar_kin)
        E_dis.append(Evar_dis)
        E_rev.append(Evar_coh)
        Ext_str.append(ext_test)
        tot_str.append(tot)
        
        damage_mean.append(np.mean((model.getMaterial("cohesive").getInternalReal("damage")(aka._cohesive_2d_4))))

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'damage_mean': damage_mean,
        'potential_energy': E_pot,
        'kinetic_energy': E_kin,
        'dissipated_energy': E_dis,
        'reversible_energy': E_rev,
        "external_work" : Ext_str,
        "total_energy" : tot_str
    }
    return results_dict

def Clip_opt_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
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
        'total_energy_verlet' : [],
        'coh_dissipation_actual': [],
        'bulk_dissipation_actual' : [],
        'total_dissipation_actual' :  []
    }

    #Initial condition
    

    functions = Functions_explicit_clip(parameters)
    solver = ExplicitSolver(functions, parameters)

    disp, acc, vel_predict, d = solver.initial_and_boundary_conditions(nodes)

    logging.basicConfig(level=logging.INFO)

    cohesive_dissipation = 0.0
    traction_at_center = 0.0
    bulk_dissipation = 0.0
    bulk_dissipation_act = 0.0
    previous_strain = 0.0
    previous_Epot = 0.0
    E_pot_test = 0.0
    previous_strain_energy = 0.0
    prev_jump = 0.0
    delta_strain_energy = 0.0
    strain_energy_actual = 0.0
    previous_strain = 0.0
    previous_stress = 0.0
    cohesive_dissipation_act = 0.0
    prev_traction = 0.0
    prev_d = np.zeros(N_elements-1)
    jump = 0.

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        logging.info("Starting time step %d", step)

        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel, acc, vel_predict)       
        
        lda = solver.compute_lda_clip(disp, lda, stress)
        
        d_previous = d.copy()
        if parameters.new_crack != 0: 
            d = solver.compute_damage_clip_gd(d, d_previous, disp)
            
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)

        d_str.append(d[math.floor((parameters.N_elements-1)/2)])

        logging.info(f"Damage at step {step}: {d}")

        stress, strain, stress_bulk, traction = solver.get_stress(disp, d, lda)

        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
     
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)
        
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        
        if parameters.new_crack != 0 :
            results_dict['nodes_str_crack'] = (solver.get_nodes())
            results_dict['displacement_crack'].append(np.copy(disp))
            results_dict['velocity_crack'].append(np.copy(vel))
            results_dict['acceleration_crack'].append(np.copy(acc))
            
        else :
            results_dict['nodes_str'] = (solver.get_nodes())
            results_dict['displacement'].append(np.copy(disp))
            results_dict['velocity'].append(np.copy(vel))
            results_dict['acceleration'].append(np.copy(acc))
            
        results_dict['stress_str'].append(np.copy(stress_bulk))
        results_dict['opening'].append(np.copy(jump))
        results_dict['damage_mean'].append(np.copy(d[math.floor((parameters.N_elements-1)/2)]))
        results_dict['bulk_damage_nodes'].append(np.copy(D_nodes))
        results_dict['GD_function'].append(np.copy(functions.GD_bulk.get_value(D_nodes)))
        results_dict['strain_str'].append(np.copy(strain))
        results_dict['dissipated_energy'].append(np.copy(energies['Edissip']))
        results_dict['dissipated_bulk_energy'].append(np.copy(energies['Edissip_bulk']))
        results_dict['dissipated_coh_energy'].append(np.copy(energies['Edissip_coh']))
        results_dict['potential_energy'].append(np.copy(energies_verlet['Epot']))
        results_dict['kinetic_energy'].append(np.copy(energies_verlet['Ekin']))
        results_dict['cohesive_energy'].append(np.copy(energies_verlet['Ecoh']))
        results_dict['external_work'].append(np.copy(energies['Ext_work']))
        results_dict['total_energy'].append(np.copy(energies['Total']))
        results_dict['total_energy_verlet'].append(np.copy(energies_verlet['Total']))

        results_dict['coh_dissipation_actual'].append(np.copy(energies_verlet['Edissip_coh']))
        results_dict['bulk_dissipation_actual'].append(np.copy(energies_verlet['Edissip_bulk']))
        results_dict['total_dissipation_actual'].append(np.copy(energies_verlet['Edissip']))

        #logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    filename = generate_filename()
    np.savez(filename, **results_dict)

    return results_dict

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
        'total_energy_verlet' : [],
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
            print("d_here", d)
            N_nodes_half = math.ceil(parameters.N_nodes / 2) 
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            traction_at_center, d[math.floor((parameters.N_elements-1)/2)], jump = solver.traction_predict(jump,d[math.floor((parameters.N_elements-1)/2)] ,opt = False)
            trac_average = (traction_at_center + prev_traction )/2
            cohesive_dissipation += trac_average * (jump - prev_jump) * parameters.Area
            prev_jump = jump
            prev_traction = traction_at_center

        D_nodes, D, _ = bulk_damage.get_Bulk_damage(d, centeronly=False)
        logging.info(f"Damage at step {step}: {d}")

        stress, strain, stress_bulk ,traction = solver.get_stress(disp, d, lda)
        logging.debug(f"Stress: {stress}")

        force = solver.get_nodal_forces(stress)
        logging.debug(f"force: {force}")
        
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        logging.debug(f"Acceleration: {acc}")

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        logging.debug(f"Velocity: {vel}")
     
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        # strain_increment = strain - previous_strain
        # stress_average = (stress_bulk + previous_stress )/2
        # bulk_dissipation_increment = parameters.dx * parameters.Area * np.dot(stress_average , strain_increment)
        # bulk_dissipation += (bulk_dissipation_increment)
        # bulk_dissipation_act = (bulk_dissipation - energies['Epot'])

        # previous_strain = strain
        # previous_stress = stress_bulk

        # cohesive_dissipation_act = (cohesive_dissipation - energies['Ecoh'])
        # total_dissipation = bulk_dissipation_act + cohesive_dissipation_act

        # Total_energy = energies['Epot'] + energies['Ekin']  + total_dissipation + energies['Ecoh']

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
        results_dict['dissipated_energy'].append(np.copy(energies['Edissip']))
        results_dict['dissipated_bulk_energy'].append(np.copy(energies['Edissip_bulk']))
        results_dict['dissipated_coh_energy'].append(np.copy(energies['Edissip_coh']))
        results_dict['potential_energy'].append(np.copy(energies_verlet['Epot']))
        results_dict['kinetic_energy'].append(np.copy(energies_verlet['Ekin']))
        results_dict['cohesive_energy'].append(np.copy(energies_verlet['Ecoh']))
        results_dict['external_work'].append(np.copy(energies['Ext_work']))
        results_dict['total_energy'].append(np.copy(energies['Total']))
        results_dict['total_energy_verlet'].append(np.copy(energies_verlet['Total']))

        results_dict['coh_dissipation_actual'].append(np.copy(energies_verlet['Edissip_coh']))
        results_dict['bulk_dissipation_actual'].append(np.copy(energies_verlet['Edissip_bulk']))
        results_dict['total_dissipation_actual'].append(np.copy(energies_verlet['Edissip']))

    filename = generate_filename()
    np.savez(filename, **results_dict)

    return results_dict

def lip_algorithm(dt, parameters):

    logging.basicConfig(level=logging.INFO)
    #logging.basicConfig(level=logging.DEBUG)

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
    jump = 0
    disp = np.zeros(N_nodes)
    vel = np.zeros(N_nodes)
    acc = np.zeros(N_nodes)
    D = np.zeros(N_nodes)
    D_prev = np.zeros(N_nodes)

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
        'total_energy_verlet' : [],
        'coh_dissipation_actual': [],
        'bulk_dissipation_actual' : [],
        'total_dissipation_actual' :  []
    }

    functions = Functions_Lip(parameters)
    solver = ExplicitSolver(functions, parameters)
    #vel = nodes * eps0dot
    
    disp, acc, vel_predict, D = solver.initial_and_boundary_conditions(nodes)
    
    D_prev = D.copy()

    vel = solver.compute_velocity(dt, vel, vel_predict, acc)
    
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
            D = solver.call_method(D, disp, D_prev)
        D_prev = D.copy()
            
        logging.info(f"Damage at step {step}: {D}")

        stress, strain = solver.get_stress_lip(disp, D)
        logging.debug(f"Stress: {stress}")

        force = solver.get_nodal_forces_lip(stress)
        logging.debug(f"force: {force}")
        
        mass = solver.get_M_lumped_lip()
        acc = solver.compute_acceleration(force, mass)
        #acc[-1] = 0.0
        logging.debug(f"Acceleration: {acc}")
        print("acceleration:",acc)

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        logging.debug(f"Velocity: {vel}")

        #D = solver.checkstress_lip(stress, D)
     
        #energies = solver.Energy_computation(disp, vel, lda, D, stress, strain, dt, Ext_work)
        energies_verlet, strain_increment = solver.Energy_verlet_lip_computation(disp, vel, vel_predict, D, stress, strain, dt, Ext_work)

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
        results_dict['damage_mean'].append(np.copy(D[math.floor((parameters.N_elements-1)/2)]))
        results_dict['bulk_damage_nodes'].append(np.copy(D))
        results_dict['GD_function'].append(np.copy(functions.GD_bulk.get_value(D)))
        results_dict['strain_str'].append(np.copy(strain))
        results_dict['dissipated_energy'].append(np.copy(energies_verlet['Edissip']))
        results_dict['dissipated_bulk_energy'].append(np.copy(energies_verlet['Edissip_bulk']))
        results_dict['dissipated_coh_energy'].append(np.copy(energies_verlet['Edissip_coh']))
        results_dict['potential_energy'].append(np.copy(energies_verlet['Epot']))
        results_dict['kinetic_energy'].append(np.copy(energies_verlet['Ekin']))
        results_dict['cohesive_energy'].append(np.copy(energies_verlet['Ecoh']))
        results_dict['external_work'].append(np.copy(energies_verlet['Ext_work']))
        results_dict['total_energy'].append(np.copy(energies_verlet['Total']))
        results_dict['total_energy_verlet'].append(np.copy(energies_verlet['Total']))

        results_dict['coh_dissipation_actual'].append(np.copy(energies_verlet['Edissip_coh']))
        results_dict['bulk_dissipation_actual'].append(np.copy(energies_verlet['Edissip_bulk']))
        results_dict['total_dissipation_actual'].append(np.copy(energies_verlet['Edissip']))

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
    eps0dot = 7
    max_steps = 100
    Dm = 0.0
    N_elements = 10
    new_crack = 0
    boundary_type = "lip"
    damage_function = 'LIP'
    nlc = 4
    
    ################################
    mesh_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/bar_40.msh"
    material_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/material.dat"

    c = np.sqrt(E/rho)
    dt = 0.1 * ((L/N_elements)/c)
    
    paramters_lip = initialize_parameters_clip()
    results_lip = lip_algorithm(dt, paramters_lip)

    boundary_type = "string"
    # paramters_akantu = initialize_parameters_clip()
    # results_aka = Akantu_algorithm(dt, material_file, mesh_file,paramters_akantu)

    # paramters_opti_clip = initialize_parameters_clip()
    # results_opti_clip = Clip_opt_algorithm(dt, paramters_opti_clip)

    # paramters_no_opti_clip = initialize_parameters_clip()
    # results_no_opti_clip = Clip_no_opt_algorithm(dt, paramters_no_opti_clip)
   
    
    # Dm = 0.6
    # paramters_opti_clip_1 = initialize_parameters_clip()
    # results_no_opti_clip_1 = Clip_no_opt_algorithm(dt, paramters_opti_clip_1)
   
    # Dm = 0.8
    # paramters_opti_clip_2 = initialize_parameters_clip()
    # results_no_opti_clip_2 = Clip_no_opt_algorithm(dt, paramters_opti_clip_2)

    # Dm = 0.9
    # paramters_opti_clip_3 = initialize_parameters_clip()
    # results_no_opti_clip_3 = Clip_no_opt_algorithm(dt, paramters_opti_clip_3)

    ###############################
    ###Post Process###
    time_index = 20
    bulk_damage_at_nodes(results_lip)
    #strain_at_timestep(dt, time_index, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2,)
    #Stress_along_bar_at_time_step(dt, time_index,results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2 )
    #energies_comparison(results_aka, results_opti_clip)
    energies_comparison_string(dt, results_lip)