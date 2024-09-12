import sys
sys.path.insert(0,"/home/ssshetty/Home/Main/Akantu/akantu/build/python")
import akantu as aka
import numpy as np
import math
from input import Explicit_Parameter
from functions import FixedDisplacement, Functions_explicit_czm
from solve import ExplicitSolver
from post_process import damage_comp_plot, energy_comp_plot
from tqdm import tqdm

def initialize_parameters():
    return Explicit_Parameter(E, Gc, sigc, rho, L, Area, eps0dot, max_steps, N_elements)


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
    functor_r = FixedDisplacement(aka._x, L*eps0dot)
    model.applyBC(functor_r, 'right')
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

    for i in tqdm(range(0, parameters.max_steps)):
        time = time + dt
        # print("i = ", i )
        # print("time = ", time)

        functor_r.set_time(time)
        model.applyBC(functor_r, 'right')

        model.dump()
        model.dump('cohesive elements')
        
        model.checkCohesiveStress()
        model.solveStep('explicit_lumped')

        E_pot.append(model.getEnergy("potential"))     
        E_kin.append(model.getEnergy("kinetic"))
        E_dis.append(model.getEnergy("dissipated"))
        E_rev.append(model.getEnergy("reversible"))
        E_con.append(model.getEnergy("contact"))

        #print("Opening =", (model.getMaterial("cohesive").getInternalReal("opening")(aka._cohesive_2d_4)))    
        damage_mean.append(np.mean((model.getMaterial("cohesive").getInternalReal("damage")(aka._cohesive_2d_4))))

    results_dict = {
        'inputs': parameters.to_dict(),
        'time_step': dt,
        'damage_mean': damage_mean,
        'potential_energy': E_pot,
        'kinetic_energy': E_kin,
        'dissipated_energy': E_dis,
    }
    return results_dict

def Czm_opti_algorithm(dt, parameters):

    N_nodes = parameters.N_nodes
    N_elements = parameters.N_elements
    sigc = parameters.sigc
    nodes = np.linspace(0, parameters.L, N_nodes)
   
    new_crack = 0
    time = 0
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
    

    #Initial conditions
    vel = nodes * eps0dot

    functions = Functions_explicit_czm()
    solver = ExplicitSolver(functions, parameters)

    for i in tqdm(range(0, max_steps)):
        
        print("")
        print("time step =", i)
        time = time + dt

        disp[0] = 0
        disp[-1]  = parameters.L * eps0dot * time
        
        d_previous = d.copy()

        vel_predict = solver.velocity_predict(dt, vel, acc)
  
        disp = solver.compute_displacement(dt, disp, vel_predict)       
        
        lda, new_crack = solver.compute_lagrange(disp, new_crack, lda, sigc, stress)

        d = solver.compute_damage(d, d_previous, lda, sigc)
        d_str.append(d[math.floor((parameters.N_elements-1)/2)])        
        print("d = ",d)

        stress, new_crack = solver.get_stress(disp, new_crack, d, lda, sigc)
        force = solver.get_nodal_forces(stress, new_crack)
        mass = solver.get_M_lumped(new_crack)
        acc = solver.compute_acceleration(force, mass)
        
        vel = solver.compute_velocity(dt,vel, vel_predict, acc)
     
        disp, vel, acc, new_crack, sigc = solver.checkcohesivestress(disp, vel, acc, new_crack, sigc, stress)
        Ep, Ekin, Edis = solver.Energy_computation(sigc, new_crack, disp, vel, lda, d)
        
        Edis_str.append(Edis)
        Ep_str.append(Ep)
        Ekin_str.append(Ekin)

    results_dict = {
            'inputs': parameters.to_dict(),
            'time_step': dt,
            'damage_mean': d_str,
            'potential_energy': Ep_str,
            'kinetic_energy': Ekin_str,
            'dissipated_energy': Edis_str,
    }

    return results_dict

   
if __name__ == "__main__" :

    L = 1
    Area = 0.1
    E = 3e9
    rho = 275
    sigc = 3e6
    Gc = 120
    eps0dot = 5
    max_steps = 200
    N_elements = 10
    
    ################################
    mesh_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/bar_1.msh"
    material_file = "/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/material.dat"

    parameters_akantu = initialize_parameters()
    results_aka = Akantu_algorithm(material_file, mesh_file, parameters_akantu )

    ###############################

    parameters_opti_czm = initialize_parameters()
    results_opti_czm = Czm_opti_algorithm(results_aka['time_step'], parameters_opti_czm)

    ###############################
    
    damage_comp_plot(results_aka['time_step'], max_steps, results_aka['damage_mean'], results_opti_czm['damage_mean'])
    energy_comp_plot(results_aka['time_step'], max_steps,
                     results_aka['potential_energy'], results_opti_czm['potential_energy'],
                     results_aka['kinetic_energy'], results_opti_czm['kinetic_energy'],
                     results_aka['dissipated_energy'], results_opti_czm['dissipated_energy'])


    