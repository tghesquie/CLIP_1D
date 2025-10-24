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
from functions import  Functions_explicit_czm,Functions_explicit_clip, Functions_Lip, Functions_4_terms
from bulk_damage import BulkDamage
from solve import ExplicitSolver, Solver, ExplicitSolverLip
from post_process import damage_comp_plot, energy_comp_plot, bulk_damage_at_nodes, strain_at_timestep, opening_along_time, GD_function_value_along_bar, velocity_along_the_bar, velocity_along_the_bar_3D, velocity_along_bar_at_timestep, displacement_along_bar_at_timestep, energies_comparison, acceleration_along_bar_at_timestep,energies_comparison_string,Stress_along_bar_at_time_step, energies_comparison_free
from tqdm import tqdm
import logging


def initialize_parameters():
    return Explicit_Parameter(E, Gc, sigc, rho, L, Area, eps0dot, max_steps, N_elements, new_crack)

def initialize_parameters_clip():
    return Clip_Explicit_Parameters(E, Gc, sigc, rho, Area, eps0dot, L, Dm, max_steps, N_elements,new_crack,boundary_type, nlc, damage_function)

def generate_filename(base):
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

    # model.applyBC(aka.FixedValue(0., aka._x), 'left')
    # model.applyBC(aka.FixedValue(0., aka._y), 'bottom')
    # model.applyBC(aka.FixedValue(0., aka._y), 'top')
    #functor_r = FixedDisplacement(aka._x, L*eps0dot)
    model.applyBC(aka.FixedValue(0., aka._x), 'left')
    model.applyBC(aka.FixedValue(0., aka._x), 'right')

    model.getExternalForce()[:] = 0

    disp_field = np.zeros(nodes.shape)
    disp_field[:, 0] = (nodes[:, 0])*(parameters.sigc )/(parameters.E) 
    model.getDisplacement()[:] = disp_field

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
        # print("vel = ", vel_test)
        # print("stress = ", stress_test)
        # print("disp = ", model.getDisplacement())
        ext_test += -((stress_test[0][0]*  vel_test[0][0] * parameters.Area * dt) + (stress_test[2][0]* vel_test[2][0] *parameters.Area * dt))

        tot = Evar_pot + Evar_kin + Evar_dis + Evar_coh
        
        E_pot.append(Evar_pot)
        E_kin.append(Evar_kin)
        E_dis.append(Evar_dis)
        E_rev.append(Evar_coh)
        Ext_str.append(ext_test)
        tot_str.append(tot)
        
        #print("Opening =", (model.getMaterial("cohesive").getInternalReal("opening")(aka._cohesive_2d_4)))    
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

def Clip_no_opt_algorithm(dt, parameters):

    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
    jump = 0
    disp = np.zeros(N_nodes)

    vel = np.zeros(N_nodes)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    lda = np.zeros(N_elements-1)

    stress = []
    method = 'Non_Opti'
    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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
    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)
    
    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        print("")
        time += dt
        logging.info(" Time %f", time)
        logging.info(" Time step %d", step)
        
        # Velocity Predictor
        vel_predict = solver.velocity_predict(dt, vel, acc)

        # Displacement Calculation
        disp = solver.compute_displacement(dt, disp, vel_predict)
        logging.debug(f"Displacement: {disp}")

        # Cohesive Damage Prediction
        if parameters.new_crack != 0:
            N_nodes_half = math.ceil(parameters.N_nodes / 2) 
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            traction_at_center, d[math.floor((parameters.N_elements-1)/2)], jump = solver.traction_predict(jump,d[math.floor((parameters.N_elements-1)/2)] ,opt = False)

        # Bulk Damage Computation 
        D_nodes, D, _ = bulk_damage.get_Bulk_damage(d, centeronly=False)
        logging.info(f"Damage at step {step}: {d}")

        # Stress Computation 
        stress, strain, stress_bulk ,traction = solver.get_stress(disp, d, lda)
        logging.debug(f"Stress: {stress}")

        # Force at nodes
        force = solver.get_nodal_forces(stress)
        logging.debug(f"force: {force}")

        # Acceleration Computation
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        acc = solver.apply_boundary_conditions(acc)
        logging.debug(f"Acceleration: {acc}")

        # Velocity Calculation
        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        logging.debug(f"Velocity: {vel}")
        print("disp = ", disp)
        print("vel = ", vel)

        # Energy Calculation
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        # Check for Cohesive zone insertion
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        # Stroing the results
        # if parameters.new_crack != 0 :
        #     results_dict['nodes_str_crack'] = (solver.get_nodes())
        #     results_dict['displacement_crack'].append(np.copy(disp))
        #     results_dict['velocity_crack'].append(np.copy(vel))
        #     results_dict['acceleration_crack'].append(np.copy(acc))
        #     results_dict['stress_crack_str'].append(np.copy(stress))
            
        # else :
        #     results_dict['nodes_str'] = (solver.get_nodes())
        #     results_dict['displacement'].append(np.copy(disp))
        #     results_dict['velocity'].append(np.copy(vel))
        #     results_dict['acceleration'].append(np.copy(acc))
        #     results_dict['stress_str'].append(np.copy(stress))
            
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

    filename = generate_filename('results_non_opt')
    np.savez(filename, **results_dict)

    return results_dict

def clip_opt_algorithm(dt, parameters):

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

    stress = []
    
    method = 'Opti'

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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

    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)

    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)

    logging.basicConfig(level=logging.INFO)

    jump = 0.

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        logging.info("Starting time step %d", step)

        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)   
        
        d_previous = d.copy()
        if parameters.new_crack != 0:
            d = solver.compute_damage_clip_gd(d, d_previous, disp)
            
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)
        
        logging.info(f"Damage at step {step}: {d[math.floor((parameters.N_elements-1)/2)]}")

        stress, strain, stress_bulk, traction = solver.get_stress(disp, d, lda)

        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        acc = solver.apply_boundary_conditions(acc)
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

        # logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    # filename = generate_filename()
    # np.savez(filename, **results_dict)

    return results_dict

def clip_opt_test_algorithm(dt, parameters, parameters_crack):
    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)
    print(N_elements)
    time = 0
    Ext_work = 0
    
    disp = np.zeros(N_nodes)
    
    vel = np.zeros(N_nodes+1)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    D = np.zeros(N_elements)
    lda = np.zeros(N_elements-1)

    stress = []
    method = 'Opti_lda'

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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

    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)

    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)
    disp_prev = disp.copy()
    logging.basicConfig(level=logging.INFO)

    jump = 0.

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        print("--------------------------------")
        logging.info("Starting time step %d", step)

        stop = False
        iteration = 0
        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)
        print(("disp =", disp[(math.floor(len(disp) / 2)-2) : (math.floor(len(disp) / 2)+2)]))
        d_prev = d.copy()


        if parameters.new_crack == 2 :
            print(" Alternating between lda and d")
            d = solver.compute_damage_clip_gd(d, d_prev, disp)
            # while iteration < 100 and not stop:
            #     d_prev = d.copy()
            #     lda_prev = lda.copy()

            #     lda = solver.compute_lda_clip(disp, lda, d, traction)
            #     print("lda = ", lda[math.floor((parameters.N_elements-1)/2)])

            #     d = solver.compute_damage_clip(d, d_prev, disp, lda)
            #     print("damage = ", d[math.floor((parameters.N_elements-1)/2)])
            #     D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)

            #     iteration += 1
                
            #     norm_lda_fun = np.linalg.norm(lda - lda_prev)/np.linalg.norm(lda)
            #     norm_d_fun = np.linalg.norm(d - d_prev)/np.linalg.norm(d)
            #     # norm_delta_fun = np.linalg.norm(solver.functional.assemble_clip_functional_4_terms_explicit_clip(d, disp, lda) - solver.functional.assemble_clip_functional_4_terms_explicit_clip(d_prev , disp_prev ,lda_prev))/np.linalg.norm(solver.functional.assemble_clip_functional_4_terms_explicit_clip(d, disp,lda))

            #     if (norm_lda_fun < 1e-5)  and (norm_d_fun < 1e-5):
            #         stop = True
            #     print("it = ",iteration,"nrm_fun = ",norm_lda_fun, norm_d_fun)
            
            print("damage = ", d[math.floor((parameters.N_elements-1)/2)])
            print("damage = ", d)
            # print("lda =", lda[math.floor((parameters.N_elements-1)/2)] )
            print("")
        
        stress, strain, stress_bulk, traction = solver.get_stress(disp, d, lda)
        print("traction =", traction)
        disp, vel, acc, lda = solver.checkcohesivestress_test(disp, vel, acc, stress, lda)
        print("lda bfr =", lda[math.floor(len(disp) / 2) - 2])

        if parameters.new_crack == 1:
            print("New cohesive zone inertion")
            parameters_crack.x = np.linspace(0., 2*parameters.dx, 3)
            parameters_crack.N_elements = 2
            solver_crack = Solver(functions, parameters_crack)
            
            disp_mid = math.floor(len(disp) / 2)
            bc1 = disp[disp_mid-2]
            bc2 = disp[disp_mid+1]
            bc = {0: bc1, 3: bc2}
            d_center = d[disp_mid-2]
            d_center = np.zeros(1)

            disp_center = disp[(disp_mid-2) : (disp_mid+2)]
            lda_center = lda[disp_mid-2]
    
            disp[(disp_mid-2) : (disp_mid+2)], lda[disp_mid-2], d[disp_mid-2], D_test = solver_crack.solve_alt_ulda_d(d_center, disp_center, lda_center, bc)
            print(("disp after insertion =", disp[(math.floor(len(disp) / 2)-2) : (math.floor(len(disp) / 2)+2)]))
            print("lda =", lda[disp_mid-2])
            print("jump =", disp[disp_mid +1] - disp[disp_mid-1])
            stress, strain, stress_bulk, traction = solver.get_lda_stress(disp, d, lda)
            vel_predict = solver.velocity_predict(dt, vel, acc)
        
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)
        
        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        acc = solver.apply_boundary_conditions(acc)
        vel = solver.compute_velocity(dt, vel, vel_predict, acc)

        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)        
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        
        if parameters.new_crack != 0 :
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

        # logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    # filename = generate_filename()
    # np.savez(filename, **results_dict)

    return results_dict

def clip_opt_lda_alt_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)
    print(N_elements)
    time = 0
    Ext_work = 0
    test = 0
    disp = np.zeros(N_nodes)
    
    vel = np.zeros(N_nodes+1)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    D = np.zeros(N_elements)
    lda = np.zeros(N_elements-1)

    stress = []
    method = 'Opti_lda'

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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

    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)

    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)
    disp_prev = disp.copy()
    logging.basicConfig(level=logging.INFO)

    jump = 0.

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        print("--------------------------------")
        logging.info("Starting time step %d", step)

        stop = False
        iteration = 0
        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)

        d_prev = d.copy()
        if parameters.new_crack == 1:
            lda = solver.compute_lda_clip_1(disp, lda, d, stress)
            print("lda = ", lda[math.floor((parameters.N_elements-1)/2)])

            d = solver.compute_damage_clip(d, d_prev, disp, lda)
            print("damage = ", d[math.floor((parameters.N_elements-1)/2)])
            print("")
    
        elif  parameters.new_crack == 2:
            

            lda = solver.compute_lda_clip_1(disp, lda, d, traction)
            d = solver.compute_damage_clip(d, d_prev, disp, lda)
            print("damage = ", d[math.floor((parameters.N_elements-1)/2)])
            print("lda =", lda[math.floor((parameters.N_elements-1)/2)] )
            print("")
        
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)

        stress, strain, stress_bulk, traction = solver.get_stress(disp, d, lda)
 
        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        acc = solver.apply_boundary_conditions(acc)
        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)        
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)


        if parameters.new_crack != 0 :
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

        # logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    # filename = generate_filename()
    # np.savez(filename, **results_dict)

    return results_dict

def clip_opt_lda_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)
    print(N_elements)
    time = 0
    Ext_work = 0
    disp = np.zeros(N_nodes)
    
    vel = np.zeros(N_nodes+1)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    D = np.zeros(N_elements)
    lda = np.zeros(N_elements-1)

    stress = []
    method = 'Opti_lda'

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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

    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)

    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)
    disp_prev = disp.copy()
    logging.basicConfig(level=logging.INFO)

    jump = 0.

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        print("--------------------------------")
        logging.info("Starting time step %d", step)

        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)

        d_prev = d.copy()
        if parameters.new_crack == 1:
            print("New cohesive zone inertion")
            lda = solver.compute_lda_clip(disp, lda, d, stress)
            d = solver.compute_damage_clip(d, d_prev, disp, lda)
            print("damage = ", d)
            print("lda = ", lda[math.floor((parameters.N_elements-1)/2)])

        elif parameters.new_crack == 2 :
            print("lda and d")

            d = solver.compute_damage_lda_clip(d, d_prev, disp)
            print("damage = ", d[math.floor((parameters.N_elements-1)/2)])
            print("")
           
        disp_prev = disp.copy()
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)
        stress, strain, stress_bulk, traction = solver.get_stress(disp, d, lda)
        
        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        acc = solver.apply_boundary_conditions(acc)

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)        
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        if parameters.new_crack != 0 :
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

        # logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    # filename = generate_filename()
    # np.savez(filename, **results_dict)

    return results_dict

def clip_opt_lda_pred_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)
    print(N_elements)
    time = 0
    Ext_work = 0
    
    disp = np.zeros(N_nodes)
    
    vel = np.zeros(N_nodes+1)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    D = np.zeros(N_elements)
    lda = np.zeros(N_elements-1)

    stress = []
    method = 'Opti_lda_pred'

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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

    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)

    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)
    disp_prev = disp.copy()
    logging.basicConfig(level=logging.INFO)

    jump = 0.

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        print("--------------------------------")
        logging.info("Starting time step %d", step)

        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)

        d_prev = d.copy()
        if parameters.new_crack == 1:
            print("New cohesive zone inertion")
            lda = solver.compute_lda_clip_1(disp, lda, d, stress)
            d = solver.compute_damage_clip(d, d_prev, disp, lda)
            print("damage = ", d)
            print("lda = ", lda[math.floor((parameters.N_elements-1)/2)])
            print("")
            
        elif parameters.new_crack == 2 :
            print("lda and d")
            d = solver.compute_damage_clip(d, d_prev, disp, lda)
            print("damage = ", d[math.floor((parameters.N_elements-1)/2)])
            # print("damage = ", d)
            lda = solver.compute_lda_clip_1(disp, lda, d, stress)
            print("lda = ", lda[math.floor((parameters.N_elements-1)/2)])      

            print("")
            
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)
        stress, strain, stress_bulk, traction = solver.get_stress(disp, d, lda)

        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()

        acc = solver.compute_acceleration(force, mass)
        acc = solver.apply_boundary_conditions(acc)

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)        
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        if parameters.new_crack != 0 :
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

        # logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    # filename = generate_filename()
    # np.savez(filename, **results_dict)

    return results_dict

def lip_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
    disp = np.zeros(N_nodes)
    vel = np.zeros(N_nodes+1)
    acc = np.zeros(N_nodes)
    D = np.zeros(N_nodes)
    lda = np.zeros(N_elements-1)

    stress = []
    
    method = 'lip'

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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
    solver = ExplicitSolverLip(functions, parameters)

    disp, acc, vel, vel_predict, D = solver.initial_and_boundary_conditions(nodes)

    logging.basicConfig(level=logging.INFO)

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        logging.info("Starting time step %d", step)

        time += dt

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)       
        
        D_previous = D.copy()
        if parameters.new_crack != 0: 
            D = solver.call_method(D, disp, D_previous)
            
        D_center = solver.get_Bulk_damage_center(D)

        logging.info(f"Damage at step {step}: {D}")

        stress, strain = solver.get_stress(disp, D_center)

        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)

        acc[0] = 0.0
        acc[-1] = 0.0

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
     
        energies = solver.Energy_computation(disp, vel, lda, D_center, stress, strain, dt, Ext_work)
        
        energies_verlet = solver.Energy_verlet_computation(disp, vel, vel_predict, D_center, stress,strain, dt, Ext_work)

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
            
        results_dict['stress_str'].append(np.copy(stress))
        results_dict['bulk_damage_nodes'].append(np.copy(D))
        results_dict['GD_function'].append(np.copy(functions.GD_bulk.get_value(D)))
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

        # logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    # filename = generate_filename()
    # np.savez(filename, **results_dict)

    return results_dict

def Clip_gen_alpha_algorithm(dt, parameters):

    logging.basicConfig(level=logging.INFO)
    #logging.basicConfig(level=logging.DEBUG)
    alpha_m = 0.5
    alpha_f = 0

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
    jump = 0
    disp = np.zeros(N_nodes)

    vel = np.zeros(N_nodes)
    acc = np.zeros(N_nodes)
    d = np.zeros(N_elements-1)
    lda = np.zeros(N_elements-1)
    disp_pred = np.zeros(N_nodes+1)

    stress = []
    method = 'gen_alpha'
    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'method': method,
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
    functions = Functions_4_terms(parameters)
    solver = ExplicitSolver(functions, parameters)
    
    disp, acc, vel, vel_predict, d = solver.initial_and_boundary_conditions(nodes)

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        print("")
        time += dt
        logging.info(" Time %f", time)
        logging.info(" Time step %d", step)
        
        # Displacement Calculation
        disp_pred[1:], vel_predict = solver.disp_vel_predict_alpha(dt, disp, vel, acc, alpha_m, alpha_f)
        logging.debug(f"Displacement: {disp}")

        d_previous = d.copy()
        # Cohesive Damage Prediction
        if parameters.new_crack != 0:
            N_nodes_half = math.ceil(parameters.N_nodes / 2) 
            jump =  abs(disp_pred[N_nodes_half-1] - disp_pred[N_nodes_half])
            traction_at_center, d[math.floor((parameters.N_elements-1)/2)], jump = solver.traction_predict(jump,d[math.floor((parameters.N_elements-1)/2)] ,opt = False)
            # d = solver.compute_damage_clip_gd(d, d_previous, disp_pred)

        # Bulk Damage Computation 
        D_nodes, D, _ = bulk_damage.get_Bulk_damage(d, centeronly=False)
        logging.info(f"Damage at step {step}: {d}")

        # Stress Computation 
        stress, strain, stress_bulk ,traction = solver.get_stress(disp_pred, d, lda)
        logging.debug(f"Stress: {stress}")

        # Force at nodes
        force = solver.get_nodal_forces(stress)
        logging.debug(f"force: {force}")

        # Acceleration Computation
        mass = solver.get_M_lumped()
        acc = solver.compute_acceleration(force, mass)
        acc[0] = 0
        acc[-1] = 0

        logging.debug(f"Acceleration: {acc}")

        # Velocity Calculation
        disp[1:], vel[1:] = solver.disp_vel_compute_alpha(dt, disp_pred, vel_predict, acc, alpha_f, alpha_m)
        logging.debug(f"Velocity: {vel}")

        # Energy Calculation
        energies = solver.Energy_computation(disp, vel, lda, d, stress, strain, dt, Ext_work)
        energies_verlet, strain_increment = solver.Energy_verlet_computation(disp, vel, vel_predict, d, stress_bulk,traction, strain, dt, Ext_work)

        # Check for Cohesive zone insertion
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        # Stroing the results
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
        results_dict['potential_energy'].append(np.copy(energies['Epot']))
        results_dict['kinetic_energy'].append(np.copy(energies['Ekin']))
        results_dict['cohesive_energy'].append(np.copy(energies['Ecoh']))
        results_dict['external_work'].append(np.copy(energies['Ext_work']))
        results_dict['total_energy'].append(np.copy(energies['Total']))
        results_dict['total_energy_verlet'].append(np.copy(energies_verlet['Total']))

        results_dict['coh_dissipation_actual'].append(np.copy(energies_verlet['Edissip_coh']))
        results_dict['bulk_dissipation_actual'].append(np.copy(energies_verlet['Edissip_bulk']))
        results_dict['total_dissipation_actual'].append(np.copy(energies_verlet['Edissip']))

    #filename = generate_filename('results_non_opt')
    #np.savez(filename, **results_dict)

    return results_dict

if __name__ == "__main__" :

    L = 1
    Area = 0.1
    E = 3e10
    rho = 275
    # Yc = 820e3
    # sigc = np.sqrt( Yc * 2 * E)
    sigc = 3e6
    Gc = 120
    eps0dot = 10
    nlc = 5
    he = 1
    N_elements = int((he * 5 * nlc))
    # max_steps = he * 1000
    # N_elements = 200
    # max_steps = int((500/40)*N_elements)
    Dm = 0.
    new_crack = 0
    boundary_type = "free"
    damage_function = 'hybrid_1'
    #check is changed
    clip_method = "Opti"

    # Time Step Computation
    c = np.sqrt(E/rho)
    dt = 0.1 * ((L/N_elements)/c)
    print(dt, N_elements)

    # dt = 5.45571e-07
    # dt =  9.09675e-08

    max_steps = int(4e-04/dt)
    max_steps = 200
    # max_steps = int(5.6e-08/dt)

    if clip_method == "No_opti":
        # Without the Optimiser
        paramters_no_opti_clip = initialize_parameters_clip()
        results_no_opti_clip = Clip_no_opt_algorithm(dt, paramters_no_opti_clip)

        # Dm = 0.4
        # paramters_no_opti_clip_1 = initialize_parameters_clip()
        # results_no_opti_clip_1 = Clip_no_opt_algorithm(dt, paramters_no_opti_clip_1)

        # Dm = 0.8
        # paramters_no_opti_clip_2 = initialize_parameters_clip()
        # results_no_opti_clip_2 = Clip_no_opt_algorithm(dt, paramters_no_opti_clip_2)

        print(results_no_opti_clip['inputs'].get('gamma'))

    elif clip_method == "Opti":

        #With the Optimiser
        paramters_opti_clip = initialize_parameters_clip()
        results_opti_clip = clip_opt_algorithm(dt, paramters_opti_clip)

        Dm = 0.4
        paramters_opti_clip_1 = initialize_parameters_clip()
        results_opti_clip_1 = clip_opt_algorithm(dt, paramters_opti_clip_1)

        Dm = 0.8
        paramters_opti_clip_2 = initialize_parameters_clip()
        results_opti_clip_2 = clip_opt_algorithm(dt, paramters_opti_clip_2)

        print(results_opti_clip['inputs'].get('gamma'))

    elif clip_method == "Opti_alt_lda":

        # With the Optimiser including lda
        paramters_opti_lda_clip = initialize_parameters_clip()
        results_opti_lda_alt_clip = clip_opt_lda_alt_algorithm(dt, paramters_opti_lda_clip)

        # Dm = 0.4
        # paramters_opti_lda_clip_1 = initialize_parameters_clip()
        # results_opti_lda_alt_clip_1 = clip_opt_lda_alt_algorithm(dt, paramters_opti_lda_clip_1)

        # Dm = 0.8
        # paramters_opti_lda_clip_2 = initialize_parameters_clip()
        # results_opti_lda_alt_clip_2 = clip_opt_lda_alt_algorithm(dt, paramters_opti_lda_clip_2)
        print(results_opti_lda_alt_clip['inputs'].get('gamma'))
    
    elif clip_method == "Opti_lda":

        # With the Optimiser including lda
        paramters_opti_lda_clip = initialize_parameters_clip()
        results_opti_lda_clip = clip_opt_lda_algorithm(dt, paramters_opti_lda_clip)

        # Dm = 0.4
        # paramters_opti_lda_clip_1 = initialize_parameters_clip()
        # results_opti_lda_clip_1 = clip_opt_lda_algorithm(dt, paramters_opti_lda_clip_1)

        # Dm = 0.8
        # paramters_opti_lda_clip_2 = initialize_parameters_clip()
        # results_opti_lda_clip_2 = clip_opt_lda_algorithm(dt, paramters_opti_lda_clip_2)

        print(results_opti_lda_clip['inputs'].get('gamma'))

    elif clip_method == "Opti_lda_pred":

        # With the Optimiser including lda
        paramters_opti_lda_clip = initialize_parameters_clip()
        results_opti_lda_pred_clip = clip_opt_lda_pred_algorithm(dt, paramters_opti_lda_clip)

        Dm = 0.4
        paramters_opti_lda_clip_1 = initialize_parameters_clip()
        results_opti_lda_pred_clip_1 = clip_opt_lda_pred_algorithm(dt, paramters_opti_lda_clip_1)

        Dm = 0.8
        paramters_opti_lda_clip_2 = initialize_parameters_clip()
        results_opti_lda_pred_clip_2 = clip_opt_lda_pred_algorithm(dt, paramters_opti_lda_clip_2)
        print(results_opti_lda_pred_clip['inputs'].get('gamma'))
    
    elif clip_method == "gen_alpha":
        paramters_gen_alpha_clip = initialize_parameters_clip()
        results_gen_alpha_clip = Clip_gen_alpha_algorithm(dt, paramters_gen_alpha_clip)

        Dm = 0.4
        paramters_gen_alpha_clip_1 = initialize_parameters_clip()
        results_gen_alpha_clip_1 = Clip_gen_alpha_algorithm(dt, paramters_gen_alpha_clip_1)

        Dm = 0.8
        paramters_gen_alpha_clip_2 = initialize_parameters_clip()
        results_gen_alpha_clip_2 = Clip_gen_alpha_algorithm(dt, paramters_gen_alpha_clip_2)

        print(results_gen_alpha_clip['inputs'].get('gamma'))

    elif clip_method == "Opti_test":

        #With the Optimiser
        paramters_opti_clip = initialize_parameters_clip()
        parameters_crack = initialize_parameters_clip()
        results_opti_clip = clip_opt_test_algorithm(dt, paramters_opti_clip, parameters_crack)

        # Dm = 0.6
        # paramters_opti_clip_1 = initialize_parameters_clip()
        # results_opti_clip_1 = clip_opt_test_algorithm(dt, paramters_opti_clip_1, parameters_crack)

        # Dm = 0.8
        # paramters_opti_clip_2 = initialize_parameters_clip()
        # results_opti_clip_2 = clip_opt_test_algorithm(dt, paramters_opti_clip_2, parameters_crack)

        print(results_opti_clip['inputs'].get('gamma'))

    ############### Akantu #################################
    mesh_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/bar_40.msh"
    material_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/material.dat"

    # parameters_akantu = initialize_parameters()
    # results_aka = Akantu_algorithm(dt, material_file, mesh_file, parameters_akantu )

    ############### LIP #################################
    # Load data from .npz file
    filename_load = "/home/ssshetty/Home/Main/Code/CLIP_1D/src/results.npz"
    data = np.load(filename_load)
    results_lip = {
        't_vect': data['t_vect'],
        'xn': data['xn'],
        'elastic_energy_vect': data['elastic_energy_vect'],
        'kinetic_energy_vect': data['kinetic_energy_vect'],
        'dissipated_energy_vect': data['dissipated_energy_vect'],
        'bulk_dissipation_actual': data['bulk_dissipation_actual'],
        'd_vect': data['d_vect'],
        'Total_energy': data['elastic_energy_vect'] + data['kinetic_energy_vect'] + data['dissipated_energy_vect'] ,
        'Total_energy_actual':data['elastic_energy_vect'] + data['kinetic_energy_vect'] + data['bulk_dissipation_actual']
    }

    # Post processing
    if boundary_type == "string":
        if clip_method == "Opti":
            energies_comparison_free(dt, results_opti_clip, results_opti_clip_1, results_opti_clip_2)

            energies_comparison_string(dt,results_lip, results_opti_clip,results_opti_clip_1,results_opti_clip_2)

        elif clip_method == "No_opti":
            energies_comparison_string(dt,results_lip, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2)
        
        elif clip_method == "Opti_alt_lda":
            energies_comparison_string(dt, results_lip, results_opti_lda_alt_clip, results_opti_lda_alt_clip_1, results_opti_lda_alt_clip_2)

        elif clip_method == "Opti_lda":
            energies_comparison_string(dt, results_lip, results_opti_lda_clip, results_opti_lda_clip_1, results_opti_lda_clip_2)
        
        elif clip_method == "Opti_lda_pred":
            energies_comparison_string(dt, results_lip, results_opti_lda_pred_clip, results_opti_lda_pred_clip_1, results_opti_lda_pred_clip_2)
        
        elif clip_method == 'gen_alpha':
            energies_comparison_string(dt,results_lip, results_gen_alpha_clip, results_gen_alpha_clip_1, results_gen_alpha_clip_2)

        elif clip_method == "Opti_test":
            energies_comparison_string(dt,results_lip,results_opti_clip)
        
    elif boundary_type == "free":
        if clip_method == "Opti":
            energies_comparison_free(dt, results_opti_clip)

        elif clip_method == "No_opti":
            energies_comparison_free(dt,results_no_opti_clip)
        
        elif clip_method == "Opti_alt_lda":
            energies_comparison_free(dt, results_opti_lda_alt_clip)

        elif clip_method == "Opti_lda":
            energies_comparison_free(dt, results_opti_lda_clip, results_opti_lda_clip_1, results_opti_lda_clip_2)
        
        elif clip_method == "Opti_lda_pred":
            energies_comparison_free(dt, results_opti_lda_pred_clip, results_opti_lda_pred_clip_1, results_opti_lda_pred_clip_2)
        
        elif clip_method == 'gen_alpha':
            energies_comparison_free(dt, results_gen_alpha_clip, results_gen_alpha_clip_1, results_gen_alpha_clip_2)

        elif clip_method == "Opti_test":
            energies_comparison_free(dt, results_opti_clip  )
        