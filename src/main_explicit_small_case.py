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
from post_process import damage_comp_plot, energy_comp_plot, bulk_damage_at_nodes, strain_at_timestep, opening_along_time, GD_function_value_along_bar, velocity_along_the_bar, velocity_along_the_bar_3D, velocity_along_bar_at_timestep, displacement_along_bar_at_timestep, energies_comparison, acceleration_along_bar_at_timestep, stress_strain_plot
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

class FixedDisplacement (aka.DirichletFunctor):
    '''
        Fix the displacement at its current value
    '''

    def __init__(self, axis, vel):
        super().__init__(axis)
        self.axis = axis
        self.time = 0
        self.vel = vel

    def set_time(self, t):
        self.time = t

    def get_imposed_disp(self):
        return self.vel*self.time
        
    def __call__(self, node, flags, disp, coord):
        # sets the blocked dofs vector to true in the desired axis
        flags[int(self.axis)] = True
        disp[int(self.axis)] = self.get_imposed_disp()


def Akantu_algorithm(material_file, mesh_file, parameters ):

    aka.parseInput(material_file)
    spatial_dimension = 2
    mesh = aka.Mesh(spatial_dimension)
    mesh.read(mesh_file)

    nodes = mesh.getNodes()

    model = aka.SolidMechanicsModelCohesive(mesh)
    model.getElementInserter().setLimit(aka._x, -0.1, 0.1)
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

    #model.applyBC(aka.FixedValue(0., aka._x), 'left')
    #model.applyBC(aka.FixedValue(0., aka._y), 'bottom')
    #model.applyBC(aka.FixedValue(0., aka._y), 'top')
    #functor_r = FixedDisplacement(aka._x, L*eps0dot)

    model.getExternalForce()[:] = 0

    vel_field = np.zeros(nodes.shape)
    vel_field[:, 0] = (nodes[:, 0])*eps0dot
    model.getVelocity()[:] = vel_field
    
    time = 0
    model.dump()
    dt = model.getStableTimeStep()*0.1
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
        print("stress = ", stress_test)
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

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(-parameters.L/2, parameters.L/2, N_nodes)

    time = 0
    Ext_work = 0
    jump = 0
    disp = np.zeros(N_nodes)
    vel = np.zeros(N_nodes)
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
    
    vel = solver.initial_and_boundary_conditions(nodes)

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

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        
        time += dt
        print("Time = ", time)
        logging.info("Starting time step %d", step)
        logging.debug("Starting time step %d", step)
        
        
        vel_predict = solver.velocity_predict(dt, vel, acc)
        
        disp = solver.compute_displacement(dt, disp, vel, acc, vel_predict)
        print("disp = ", disp)

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
        print("stress = ", stress)
        nodal_forces = solver.get_nodal_forces(stress)        
        mass = solver.get_M_lumped()

        acc = solver.compute_acceleration(nodal_forces, mass)
        print("acc =", acc)
        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        print("vel =", vel)
        # vel[0] = eps0dot*-parameters.L/2
        # vel[-1] = eps0dot*parameters.L/2
     
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)
   
        Ep, Ekin, Edis, Ext_work, Ecoh, Total_energy,Edissip_bulk, Edissip_coh  = solver.Energy_computation(disp, vel, lda, d, stress, dt,Ext_work)
        
   
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


def compute_stiffness_matrix(N_elements, E, A, L_total):
    """
    Compute the global stiffness matrix for a 1D bar using FEM.
    
    Parameters:
    N_elements : int
        The number of elements in the bar.
    E : float
        Young's modulus of the material.
    A : float
        Cross-sectional area of the bar.
    L_total : float
        Total length of the bar.
    
    Returns:
    K_global : 2D numpy array
        The global stiffness matrix of the bar.
    """
    
    # Number of nodes
    N_nodes = N_elements + 1
    
    # Length of each element
    L_element = L_total / N_elements
    
    # Initialize global stiffness matrix
    K_global = np.zeros((N_nodes, N_nodes))
    
    # Element stiffness matrix (same for all elements in a uniform bar)
    K_element = (E * A / L_element) * np.array([[1, -1], [-1, 1]])
    
    # Assembly of the global stiffness matrix
    for e in range(N_elements):
        # Global node numbers for the current element
        n1 = e       # Left node of element
        n2 = e + 1   # Right node of element
        
        # Add element stiffness matrix to the global matrix
        K_global[n1:n2+1, n1:n2+1] += K_element
    
    return K_global
if __name__ == "__main__" :

    L = 1
    Area = 0.1
    E = 3e9
    rho = 275
    sigc = 3e6
    Gc = 120
    eps0dot = 10
    max_steps = 150
    Dm = 0.
    N_elements = 2
    new_crack = 0
    boundary_type = "small_case"
    nlc = 4
    
    ################################
    mesh_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/small_case_2.msh"
    material_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/material.dat"

    parameters_akantu = initialize_parameters()
    results_aka = Akantu_algorithm(material_file, mesh_file, parameters_akantu )

    ###############################

    # parameters_opti_czm = initialize_parameters_clip()
    # results_opti_czm = Czm_opti_algorithm(results_aka['time_step'], parameters_opti_czm)

    ###############################

    paramters_opti_clip = initialize_parameters_clip()
    # results_opti_clip = Clip_opt_algorithm(results_aka['time_step'], paramters_opti_clip)
    # c = np.sqrt(E/rho)
    # dt = 0.1 * ((L/N_elements)/c)
    
    results_no_opti_clip = Clip_no_opt_algorithm(results_aka['time_step'], paramters_opti_clip)
    #results_no_opti_clip = Clip_no_opt_algorithm(dt, paramters_opti_clip)

    # # ###############################
    Dm = 0.5
    paramters_opti_clip_1 = initialize_parameters_clip()
    results_no_opti_clip_1 = Clip_no_opt_algorithm(results_aka['time_step'], paramters_opti_clip_1)
    
    Dm = 0.8
    paramters_opti_clip_2 = initialize_parameters_clip()
    results_no_opti_clip_2 = Clip_no_opt_algorithm(results_aka['time_step'], paramters_opti_clip_2)
    
    # ###############################
    ###Post Process###
    energies_comparison(results_aka,results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2)
    #energies_comparison(results_aka, results_no_opti_clip)
    
    
    # velocity_along_the_bar(results_no_opti_clip)
    # velocity_along_the_bar_3D(results_aka['time_step'], results_no_opti_clip )
    time_index = 20
    column_index = 217

    # displacement_along_bar_at_timestep(results_aka['time_step'], time_index, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2 )
    # velocity_along_bar_at_timestep(results_aka['time_step'], time_index, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2 )
    # acceleration_along_bar_at_timestep(results_aka['time_step'], time_index, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2 )
    # opening_along_time(results_aka['time_step'], results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2)
    
    stress_strain_plot(results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2)
    GD_function_value_along_bar(results_aka['time_step'], column_index, results_no_opti_clip, results_no_opti_clip_1, results_no_opti_clip_2,)
    strain_at_timestep(results_aka['time_step'], column_index, results_no_opti_clip, results_no_opti_clip_1)
    bulk_damage_at_nodes(results_no_opti_clip_1)
    # damage_comp_plot(results_aka['time_step'], max_steps, results_aka['damage_mean'], results_no_opti_clip['damage_mean'])
    # energy_comp_plot( max_steps,results_aka, results_no_opti_clip_2)