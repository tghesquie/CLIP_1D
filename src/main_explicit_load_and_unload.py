import sys
sys.path.insert(0,"/home/ssshetty/Home/Main/Akantu/akantu/build/python")
import akantu as aka
import numpy as np
import uuid
import math
import matplotlib.pyplot as plt
from input import Explicit_Parameter, Clip_Explicit_Parameters
from functions import  Functions_explicit_czm,Functions_explicit_clip
from solve import ExplicitSolver, Solver
from post_process import damage_comp_plot, energy_comp_plot, tract_sep_law
from tqdm import tqdm
import logging

def initialize_parameters():
    return Explicit_Parameter(E, Gc, sigc, rho, L, Area, eps0dot, max_steps, N_elements, new_crack)

def initialize_parameters_clip():
    return Clip_Explicit_Parameters(E, Gc, sigc, rho, Area, eps0dot, L, Dm, max_steps, N_elements,new_crack,boundary_type)

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
    functor_r = FixedDisplacement(aka._x, L*eps0dot)

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
    Ext_str = []
    ext_test = 0
    tot_str = []

    for i in tqdm(range(0, parameters.max_steps)):
        time = time + dt
        print("i = ", i )
        print("time = ", time)

        d_trial = np.mean((model.getMaterial("cohesive").getInternalReal("damage")(aka._cohesive_2d_4)))
        print(d_trial)
        
        if d_trial > 0.1 :
            if t > 1 :
                functor_r.set_time(time -  0.76*time_pret)
                model.applyBC(functor_r, 'right')
            else:
                t+=1
                functor_r.set_time(1.2*time_pret - time )
                model.applyBC(functor_r, 'right')

        else:
            functor_r.set_time(time)
            model.applyBC(functor_r, 'right')
            time_pret = time
        
            
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
        ext_test += -(stress_test[0][0]*  vel_test[0][0] * parameters.Area * dt)

        tot = Evar_pot + Evar_kin + Evar_dis + Evar_coh + ext_test
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

def Czm_opti_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)
   
    time = 0
    Ext_work = 0
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

    #Initial condition
    vel = nodes * eps0dot

    functions = Functions_explicit_clip(parameters)
    solver = ExplicitSolver(functions, parameters)
    init_energy = solver.init_Energy_compute(disp, vel, lda, d, dt)

    logging.basicConfig(level=logging.INFO)

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        logging.info("Starting time step %d", step)

        time += dt

       
        disp[0] = 0
        disp[-1]  = parameters.L * eps0dot * time
        
        d_previous = d.copy()

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)       
        
        lda = solver.compute_lagrange(disp, lda, stress)

        d = solver.compute_damage(d, d_previous, lda)

        d_str.append(d[math.floor((parameters.N_elements-1)/2)])        
        logging.info(f"Damage at step {step}: {d}")

        stress = solver.get_stress(disp, d, lda)

        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()

        acc = solver.compute_acceleration(force, mass)        
        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
     
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)
        Ep, Ekin, Edis, Ext_work, Ecoh, Total_energy = solver.Energy_computation(disp, vel, lda, d, stress, dt,Ext_work, init_energy)

        Edis_str.append(Edis)
        Ep_str.append(Ep)
        Ekin_str.append(Ekin)
        Ecoh_str.append(Ecoh)
        Ext_work_str.append(Ext_work)
        Tot_str.append(Total_energy)

        logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    results_dict = {
            'inputs': parameters.to_dict(),
            'time_step': dt,
            'damage_mean': d_str,
            'potential_energy': Ep_str,
            'kinetic_energy': Ekin_str,
            'dissipated_energy': Edis_str,
            'cohesive_energy': Ecoh_str,
            'external_work': Ext_work_str,
            'total_energy' : Tot_str
    }

    filename = generate_filename()
    np.savez(filename, **results_dict)

    return results_dict

def Clip_opt_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
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

    #Initial condition
    vel = nodes * eps0dot

    functions = Functions_explicit_clip(parameters)
    solver = ExplicitSolver(functions, parameters)

    init_energy = solver.init_Energy_compute(disp, vel, lda, d, dt)

    logging.basicConfig(level=logging.INFO)

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        
        logging.info("Starting time step %d", step)

        time += dt

        disp[0] = 0
        disp[-1]  = parameters.L * eps0dot * time
    
        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)       
        
        lda = solver.compute_lda_clip(disp, lda, stress)
        d_previous = d.copy()
        d = solver.compute_damage_clip(d, d_previous,disp, lda)
        D_nodes, D, D_overall = solver.bulk_damage.get_Bulk_damage(d,centeronly = False)

        d_str.append(d[math.floor((parameters.N_elements-1)/2)])

        logging.info(f"Damage at step {step}: {d}")

        stress = solver.get_stress(disp, d, lda)

        force = solver.get_nodal_forces(stress)
        mass = solver.get_M_lumped()

        acc = solver.compute_acceleration(force, mass)

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
     
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        Ep, Ekin, Edis, Ext_work, Ecoh, Total_energy = solver.Energy_computation(disp, vel, lda, d, stress, dt,Ext_work, init_energy)

        Edis_str.append(Edis)
        Ep_str.append(Ep)
        Ekin_str.append(Ekin)
        Ecoh_str.append(Ecoh)
        Ext_work_str.append(Ext_work)
        Tot_str.append(Total_energy)

        logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    results_dict = {
            'inputs': parameters.to_dict(),
            'time_step': dt,
            'damage_mean': d_str,
            'potential_energy': Ep_str,
            'kinetic_energy': Ekin_str,
            'dissipated_energy': Edis_str,
            'cohesive_energy': Ecoh_str,
            'external_work': Ext_work_str,
            'total_energy' : Tot_str
    }

    filename = generate_filename()
    np.savez(filename, **results_dict)

    return results_dict

def Clip_no_opt_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    nodes = np.linspace(0, parameters.L, N_nodes)

    time = 0
    Ext_work = 0
    tract_pred = 0
    jump = 0
    jump_max = 0
    t = 0
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
    tract_pred_str = []
    jump_str = []
    
    functions = Functions_explicit_clip(parameters)
    solver = ExplicitSolver(functions, parameters)

    vel = solver.initial_and_boundary_conditions(nodes)

    logging.basicConfig(level=logging.INFO)

    for step in tqdm(range(0, max_steps), desc = "Time steps"):
        print("")
        
        time += dt
        print("Time = ", time)
        #logging.info("Starting time step %d", time)
        if d[math.floor((parameters.N_elements-1)/2)] < 0.1 :
            disp[0] = 0
            disp[-1]  = parameters.L * eps0dot * time
            time_pret = time
        elif t > 1:
            disp[0] = 0
            disp[-1]  = parameters.L * eps0dot * (time -  0.76*time_pret)

        else:
            t+=1
            disp[0] = 0
            disp[-1]  = parameters.L * eps0dot * (1.2*time_pret - time )

        

        #disp = solver.boundary_conditions_on_time(time, disp)

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)
        print("disp = ", disp)   
        
        if parameters.new_crack != 0:
            N_nodes_half = math.ceil(parameters.N_nodes / 2) 
            jump =  (disp[N_nodes_half] - disp[N_nodes_half-1] )
            tract_pred, d[math.floor((parameters.N_elements-1)/2)], jump = solver.traction_predict(jump, opt = False)              

        d_str.append(d[math.floor((parameters.N_elements-1)/2)])
        tract_pred_str.append(tract_pred)
        jump_str.append(jump)
        print("jump =", jump)
        

        logging.info(f"Damage at step {step}: {d}")

        stress = solver.get_stress(disp, d, lda)
        print("stress = ", stress)
       
        force = solver.get_nodal_forces(stress)
        print("force =", force)

        mass = solver.get_M_lumped()

        acc = solver.compute_acceleration(force, mass)
        print("acc =", acc)

        vel = solver.compute_velocity(dt, vel, vel_predict, acc)
        print("vel =",vel)
     
        disp, vel, acc = solver.checkcohesivestress(disp, vel, acc, stress)

        Ep, Ekin, Edis, Ext_work, Ecoh, Total_energy = solver.Energy_computation(disp, vel, lda, d, stress, dt,Ext_work)

        Edis_str.append(Edis)
        Ep_str.append(Ep)
        Ekin_str.append(Ekin)
        Ecoh_str.append(Ecoh)
        Ext_work_str.append(Ext_work)
        Tot_str.append(Total_energy)

        logging.info(f"Energy at step {step} - Potential: {Ep}, Kinetic: {Ekin}, Dissipated: {Edis}, External Work: {Ext_work}")

    results_dict = {
            'inputs': parameters.to_dict(),
            'time_step': dt,
            'damage_mean': d_str,
            'opening': jump_str,
            'traction': tract_pred_str,
            'potential_energy': Ep_str,
            'kinetic_energy': Ekin_str,
            'dissipated_energy': Edis_str,
            'cohesive_energy': Ecoh_str,
            'external_work': Ext_work_str,
            'total_energy' : Tot_str
    }

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
    max_steps = 27
    Dm = 0.5
    N_elements = 2
    new_crack = 0
    boundary_type = "imposed"
    
    ################################
    mesh_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/bar_2.msh"
    material_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/material.dat"

    # parameters_akantu = initialize_parameters()
    # results_aka = Akantu_algorithm(material_file, mesh_file, parameters_akantu )

    ###############################

    # parameters_opti_czm = initialize_parameters_clip()
    # results_opti_czm = Czm_opti_algorithm(results_aka['time_step'], parameters_opti_czm)

    ###############################

    paramters_opti_clip = initialize_parameters_clip()
    #results_opti_clip = Clip_opt_algorithm(results_aka['time_step'], paramters_opti_clip)
    c = np.sqrt(E/rho)
    dt = 0.1 * ((L/N_elements)/c)
    #results_no_opti_clip = Clip_no_opt_algorithm(results_aka['time_step'], paramters_opti_clip)
    results_no_opti_clip = Clip_no_opt_algorithm(dt, paramters_opti_clip)
    
    ###############################
    ###Post Process###
    
    tract_sep_law(results_no_opti_clip)
    damage_comp_plot(results_aka['time_step'], max_steps, results_aka['damage_mean'], results_no_opti_clip['damage_mean'])
    energy_comp_plot( max_steps,results_aka, results_no_opti_clip)