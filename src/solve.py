import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bulk_damage import BulkDamage

class EquilibriumSolver:
    """
    Solves the equilibrium equations for a cohesive and bulk damage simulation.
    Manages the construction and manipulation of stiffness matrices and force vectors.
    """
    def __init__(self, model, simulation_parameters):
        """
        Initializes the solver with material and simulation parameters.
        """
        self.model = model
        self.params = simulation_parameters

        # Initialize matrices to None; they will be constructed as needed.
        self.K_uu_b = None
        self.K_uu_j = None
        self.K_ul = None
        self.K_ll = None
        self.K_bc = None
        self.K = None
        self.F = None


    def get_K_uu_b(self, D_center):
        """
        Constructs or updates the bulk part of the stiffness matrix based on D_center.
        """
        # Calculate element stiffness contributions based on damage and material properties
        
        element_stiffness_contributions = (self.model.GD_bulk.get_value(D_center)[:, np.newaxis] * (self.params.E / self.params.dx * np.array([1., -1., -1., 1.]))).flatten()
        
        # Check if the bulk stiffness matrix already exists
        if self.K_uu_b is None:
            # Create row and column indices for the COO matrix format
            col_indices = np.repeat(np.arange(0, (2 * self.params.N_elements)).reshape((-1, 2)), 2, axis=0).flatten()
            row_indices = np.repeat(np.arange(0, (2 * self.params.N_elements)), 2)
            # Construct the sparse COO format stiffness matrix
            self.K_uu_b = scipy.sparse.coo_matrix((element_stiffness_contributions, (row_indices, col_indices)), shape=(2 * self.params.N_elements, 2 * self.params.N_elements))
        else:
            # Update existing matrix data if the matrix already exists
            self.K_uu_b.data = element_stiffness_contributions
        
        return self.K_uu_b


    def get_K_uu_j(self):
        """
        Constructs or updates the joint stiffness matrix.
        """
        # Check if the joint stiffness matrix already exists
        if self.K_uu_j is None:
            # Define the pattern for stiffness contributions between nodes
            stiffness_pattern = np.array([[ -self.params.k, self.params.k, self.params.k, -self.params.k]])
            
            # Repeat the stiffness pattern for each element minus one and flatten the array to use it in a sparse matrix
            data = np.repeat(stiffness_pattern, self.params.N_elements - 1, axis=0).flatten()
            
            # Generate column and row indices for COO matrix format
            col_indices = np.repeat(np.arange(1, (2 * self.params.N_elements - 1)).reshape((-1, 2)), 2, axis=0).flatten()
            row_indices = np.repeat(np.arange(1, (2 * self.params.N_elements) - 1), 2)
            
            # Create the sparse COO format joint stiffness matrix with the defined patterns
            self.K_uu_j = scipy.sparse.coo_matrix((data, (row_indices, col_indices)),
                                                shape=(2 * self.params.N_elements, 2 * self.params.N_elements))
            
        return self.K_uu_j
    

    def get_K_ul(self):
        """
        Constructs or updates the coupling stiffness matrix (K_ul) between displacement variables
        and Lagrange multipliers. 

        The matrix pattern [-1, 1] represents the influence of a unit Lagrange multiplier on
        adjacent nodal displacements, typically enforcing a difference or equality constraint
        between these nodes.
        """
        # Ensure the matrix is constructed only once unless updates are required
        if self.K_ul is None:
            # Row indices correspond to the constraints or Lagrange multipliers, each affecting a pair of adjacent nodes or elements in the simulation mesh
            row_indices = np.repeat(np.arange(self.params.N_elements - 1), 2)
            col_indices = np.arange(1, (2 * self.params.N_elements - 1))
            
            # Data entries [-1, 1] for each constraint
            data = np.repeat([[-1, 1]], self.params.N_elements - 1, axis=0).flatten()
            
            # Create the sparse COO format coupling stiffness matrix with the specified pattern
            # The shape accounts for the Lagrange multipliers (rows) and displacement DOFs (columns)
            self.K_ul = scipy.sparse.coo_matrix((data, (row_indices, col_indices)),
                                                shape=(self.params.N_elements - 1, 2 * self.params.N_elements))
        
        # Return the updated or newly created coupling stiffness matrix
        return self.K_ul


    def get_K_ll(self, d):
        """
        Updates or constructs the Lagrange-Lagrange interaction matrix K_ll.
        """
        # Compute matrix values based on damage and stiffness parameter
       
        data = -(self.model.gd_cohesive.get_lmb_value(d)) / self.params.k
        
        # Create or update the K_ll matrix
        if self.K_ll is None:
            # Generate indices for creating a square matrix
            indices = np.arange(len(data))
            # Construct a sparse matrix with damage-derived data
            self.K_ll = scipy.sparse.coo_matrix((data, (indices, indices)), shape=(len(data), len(data)))
        else:
            # Update existing matrix with new data
            self.K_ll.data = data
        
        return self.K_ll

    
    def get_K_bc(self, imposed_displacements_keys):
        """
        Constructs or updates the boundary conditions matrix K_bc.
        """
        # Initialize K_bc matrix if it hasn't been created
        if self.K_bc is None:
            # Number of boundary conditions
            nc = len(imposed_displacements_keys)
            # Row indices for boundary conditions in the global matrix
            I = np.array(imposed_displacements_keys)
            # Column indices - one per boundary condition
            J = np.arange(nc)
            # Value of 1 for each boundary condition's contribution
            Bval = np.ones(nc, dtype='float')
            # Create the boundary condition matrix
            self.K_bc = scipy.sparse.coo_matrix((Bval, (I, J)), shape=(2 * self.params.N_elements, nc))
        
        return self.K_bc


    def get_K(self, d, D_center, imposed_displacements):
        """
        Assembles the global stiffness matrix by combining the stiffness matrices
        for bulk (K_uu_b), joints (K_uu_j), coupling (K_ul), Lagrange-Lagrange interactions (K_ll),
        and boundary conditions (K_bc).
        """
        # Retrieve or update the bulk and joint stiffness matrices based on current damage and displacement
        K_uu_b = self.get_K_uu_b(D_center)
        K_uu_j = self.get_K_uu_j()
        
        # Update or retrieve matrices for coupling and boundary conditions
        K_ul = self.get_K_ul()
        K_bc = self.get_K_bc(list(imposed_displacements.keys()))
        
        # Retrieve or update the Lagrange-Lagrange interaction matrix
        K_ll = self.get_K_ll(d)
        
        # If the global stiffness matrix K already exists, update its data
        if self.K is None:

            # Assemble data for the new global stiffness matrix
            data = np.hstack([K_uu_b.data, K_uu_j.data, K_ul.data, K_bc.data, 
                              K_ul.data, K_ll.data,
                              K_bc.data])
            row_indices = np.hstack([K_uu_b.row, K_uu_j.row,  K_ul.col, K_bc.row, 
                                     K_ul.row + 2*self.params.N_elements, K_ll.row + 2*self.params.N_elements, 
                                     K_bc.col + 2*self.params.N_elements + self.params.N_elements - 1])
            col_indices = np.hstack([K_uu_b.col, K_uu_j.col, K_ul.row+ 2 *self.params.N_elements, K_bc.col + 2*self.params.N_elements + self.params.N_elements -1, 
                                     K_ul.col, K_ll.col + 2*self.params.N_elements, 
                                     K_bc.row])

            size = 2 * self.params.N_elements + (self.params.N_elements - 1) + len(imposed_displacements.keys())
            
            # Create the global stiffness matrix
            self.K = scipy.sparse.coo_matrix((data, (row_indices, col_indices)), shape=(size, size))
            
            # Save indices for efficiently updating K_ll data in future calls
            self.start_K_ll = self.K_uu_b.nnz + self.K_uu_j.nnz + 2*self.K_ul.nnz + self.K_bc.nnz
            self.end_K_ll = self.start_K_ll + self.K_ll.nnz

        else:
            self.K.data[:self.K_uu_b.nnz] = K_uu_b.data
            self.K.data[self.start_K_ll:self.end_K_ll] = K_ll.data

           
        return self.K


    def F_u_vector(self, imposed_displacements):
        """
        Updates the force vector F with values from imposed displacements.
        """
        # Determine the size of the force vector if it has not been initialized
        if self.F is None:
            self.F = np.zeros(3 * self.params.N_elements - 1 + len(imposed_displacements))
        
        # Update the portion of the force vector corresponding to imposed displacements
        # The last segment of the vector is set based on the values of the imposed displacements
        end_index = 3 * self.params.N_elements - 1
        imposed_values = np.array(list(imposed_displacements.values()))
        self.F[end_index:] = imposed_values
        
        return self.F


    def solve_equilibrium_ul(self, d, D_center, imposed_displacements):
        """
        Solves the equilibrium equations for the given damage state and imposed displacements.
        
        :param d: Array of damage values for elements.
        :param D_center: Damage state at the center of elements.
        :param imposed_displacements: Dictionary mapping node indices to their imposed displacements.
        :return: Tuple containing the displacement vector (resx), Lagrange multipliers for boundaries (resL_b),
                and Lagrange multipliers for nodes (resL_n), along with the global stiffness matrix (K).
        """
        # Assemble the global stiffness matrix for the current state
        K = self.get_K(d, D_center, imposed_displacements)
        
        # Construct the global force vector based on imposed displacements
        F = self.F_u_vector(imposed_displacements)
        
        # Solve the linear system to find displacements and Lagrange multipliers
        u = scipy.sparse.linalg.spsolve(K, F)
        
        # Extract displacements and Lagrange multipliers from the solution vector
        # Displacements at nodes
        resx = u[:(2 * self.params.N_elements)]
        # Lagrange multipliers for nodes (enforcing constraints between nodes)
        resL_n = u[(2 * self.params.N_elements):(2 * self.params.N_elements + self.params.N_elements - 1)]
        # Lagrange multipliers for boundaries (enforcing boundary conditions)
        resL_b = -u[(2 * self.params.N_elements + self.params.N_elements - 1):]
        
        return resx, resL_b, resL_n, K

class EquilibriumSolver_Lip:
    """
    Solves the equilibrium equations for the LIP model.
    Manages the construction and manipulation of stiffness matrices and force vectors.
    """

    def __init__(self,model, simulation_parameters):
        """
        Initializes the solver with material and simulation parameters.
        """
        self.model = model
        self.params = simulation_parameters
    
    def get_B(self,):
        return -1./self.params.dx*scipy.sparse.eye(self.params.N_v-1,self.params.N_v) +1./self.params.dx*scipy.sparse.eye(self.params.N_v-1,self.params.N_v,1)
    
    def d2e_eps_deps2(self, d) : 
        return self.params.dx*scipy.sparse.diags(self.params.E*self.model.GD_bulk.get_value(d))

    def K_uu(self,d):
        B = self.get_B()
        return B.T.dot(self.d2e_eps_deps2(d).dot(B))
    
    def solve_equilibrium_ul(self, d, imposed_displacements):
        Kuu = self.K_uu(d)
        n  = Kuu.shape[0]
        Fu = scipy.sparse.coo_matrix((n,1), dtype = float)

        nc = len(imposed_displacements)
        I =    np.array( list(imposed_displacements.keys()))
        J =    np.array( list(range(nc)))
        Bval = np.ones(nc, dtype ='float')
        Kul = scipy.sparse.coo_matrix((Bval, (I,J)), shape = (n,nc))
        K  = scipy.sparse.bmat([[Kuu, Kul],[Kul.T, None]], format='csr')            
        Fl = np.array([ [v] for v in imposed_displacements.values()], dtype ='float')
        F  = scipy.sparse.vstack([Fu, Fl], format ='csr' )

        u  = scipy.sparse.linalg.spsolve(K,F)
        res = {'x': u[:n], 'L': -u[n:], 'nit': 1}
        return res['x'], res['L']

class Functional:
    def __init__(self, model, simulation_parameters, bulk_damage):
        """
        Initializes the ClipFunctional with a model for material behaviors and simulation parameters.
        
        :param model: An instance of the Model class.
        :param simulation_parameters: An instance containing simulation parameters.
        """
        self.model = model
        self.params = simulation_parameters
        self.bulk_damage = bulk_damage

    def get_strain(self, u):
        """
        Calculates the strain for a given displacement field.
        """
        # Calculate the strain based on the displacement field
        return (u[1::2] - u[:-1:2]) / self.params.dx
    
    def get_strain_lip(self,u) :
        """
        Calculates the strain for a given displacement field.
        """
        return (u[1:] - u[:-1])/ self.params.dx
    
    def get_jump(self, u):
        """
        Calculates the jump in displacement for a given displacement field.
        """
        # Calculate the jump in displacement based on the displacement field
        return u[2:-1:2] - u[1:-2:2]

    def get_strain_energy(self, strain, D_center):
        """
        Calculates the strain energy for the given strain and damage state.
        """
        return 0.5 * self.params.dx * self.params.E * (self.model.GD_bulk.get_value(D_center).dot(strain**2))
    
    def get_strain_energy_derivative(self, strain, D_center):
        """
        Calculates the derivative of strain energy with respect to strain.
        """
        return 0.5 * self.params.dx * self.params.E * (self.model.GD_bulk.get_first_derivative(D_center) * (strain**2))
    
    def get_cohesive_energy(self, u_jump, d):
        """
        Calculates the cohesive energy for the given jump in displacement and damage state.
        """
        return 0.5 * self.params.k * (self.model.gd_cohesive.get_value(d).dot(u_jump**2))
        
    def get_cohesive_energy_derivative(self, u_jump, d):
        """
        Calculates the derivative of cohesive energy with respect to jump in displacement.
        """
        return 0.5 * self.params.k * (self.model.gd_cohesive.get_first_derivative(d) * (u_jump**2))
    
    def get_cohesive_dissipation(self, d):
        """
        Calculates the cohesive energy dissipation for the given damage state.
        """
        return np.sum(self.params.yc * self.model.hd_cohesive.get_value(d))
    
    def get_cohesive_dissipation_derivative(self, d):
        """
        Calculates the derivative of cohesive energy dissipation with respect to damage.
        """
        return self.params.yc * self.model.hd_cohesive.get_first_derivative(d)
    
    def get_bulk_dissipation(self, D_center):
        """
        Calculates the bulk energy dissipation for the given damage state.
        """
        return np.sum(self.params.Yc * self.params.dx * self.model.HD_bulk.get_value(D_center))
    
    def get_bulk_dissipation_derivative(self, D_center):
        """
        Calculates the derivative of bulk energy dissipation with respect to damage.
        """
        return self.params.Yc * self.params.dx * self.model.HD_bulk.get_first_derivative(D_center)
    
    def get_cohesive_energy_lagrange(self, u_jump, lambda_, d):
        """
        Calculates the cohesive energy when using Lagrange multipliers in the formulation.
        """
        term1 = 0.5 * self.params.k * np.sum(u_jump**2)
        term2 = np.dot(lambda_, u_jump)
        term3 = 1/(2 * self.params.k) * (self.model.gd_cohesive.get_lmb_value(d)).dot(lambda_**2)
        return term1, term2, term3
    
    def get_cohesive_energy_lagrange_derivative(self, lambda_,d):
        return 1/(2 * self.params.k) * (lambda_**2) * (self.model.gd_cohesive.get_derivative_lmb_value(d))

    def assemble_clip_functional_4_terms(self, d, D_center, u, lambda_,returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain(u)
        u_jump = self.get_jump(u)

        strain_energy = self.get_strain_energy(strain, D_center)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)

        clip_functional = strain_energy - ce_term1 + ce_term2 - ce_term3 + bulk_dissipation + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, ce_term1, ce_term2, ce_term3, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional

    def get_strain_explicit_clip(self, disp):
        dx =self.params.dx
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
        if self.params.new_crack == 0:
            strain = (disp[1:] - disp[:-1])/dx
            return strain

        else :
            disp_left = disp[: N_nodes_half]
            disp_right = disp[N_nodes_half: ]

            strain_left = (disp_left[1:] - disp_left[:-1])/dx
            strain_right = (disp_right[1:] - disp_right[:-1])/dx
            strain = np.concatenate((strain_left, strain_right))
            return strain

    def get_jump_explcit_clip(self, disp):
        jump = np.zeros(self.params.N_elements-1)
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
        if self.params.new_crack == 0:
            return jump 

        else:
            jump[math.floor(self.params.N_elements / 2)-1] =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            return jump
     
    def assemble_clip_functional_4_terms_explicit_clip(self, d, D_center, u, lambda_,returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain_explicit_clip(u)
        u_jump = self.get_jump_explcit_clip(u)

        strain_energy = self.get_strain_energy(strain, D_center)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)

        clip_functional = strain_energy - ce_term1 + ce_term2 - ce_term3 + bulk_dissipation + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, ce_term1, ce_term2, ce_term3, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional
        
    def assemble_jac_clip_functional_4_terms(self, d, D_center, u, lambda_, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """
        strain = self.get_strain(u)
        
        d_strain_energy = self.get_strain_energy_derivative(strain, D_center)
        d_ce_term3 = self.get_cohesive_energy_lagrange_derivative(lambda_,d)
        d_bulk_dissipation = self.get_bulk_dissipation_derivative(D_center)
        d_cohesive_dissipation = self.get_cohesive_dissipation_derivative(d)
        dD_center = self.bulk_damage.get_dBulk_damage_dd(d)

        jac_clip_functional = dD_center.T.dot(d_strain_energy + d_bulk_dissipation) - d_ce_term3 + d_cohesive_dissipation
        if returnall:
            return jac_clip_functional, d_strain_energy, d_ce_term3, d_bulk_dissipation, d_cohesive_dissipation
        else:
            return jac_clip_functional
        
    def assemble_clip_functional_3_terms(self, d, D_center, u, lambda_, returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain(u)
        u_jump = self.get_jump(u)

        strain_energy = self.get_strain_energy(strain, D_center)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)

        clip_functional = strain_energy - ce_term1 + ce_term2 - ce_term3  + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, ce_term1, ce_term2, ce_term3, cohesive_dissipation
        else:
            return clip_functional
        
    def assemble_jac_clip_functional_3_terms(self, d, D_center, u, lambda_, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """
        strain = self.get_strain(u)
        
        d_strain_energy = self.get_strain_energy_derivative(strain, D_center)
        d_ce_term3 = self.get_cohesive_energy_lagrange_derivative(lambda_,d)
        d_cohesive_dissipation = self.get_cohesive_dissipation_derivative(d)
        dD_center = self.bulk_damage.get_dBulk_damage_dd(d)

        jac_clip_functional = dD_center.T.dot(d_strain_energy) - d_ce_term3 + d_cohesive_dissipation
        if returnall:
            return jac_clip_functional, d_strain_energy, d_ce_term3, d_cohesive_dissipation
        else:
            return jac_clip_functional
        
    def assemble_czm_functional(self, d,D_pseudo_czm, u, lambda_, returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain(u)
        u_jump = self.get_jump(u)

        strain_energy = self.get_strain_energy(strain,D_pseudo_czm)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)
       

        czm_functional = strain_energy - ce_term1 + ce_term2 - ce_term3  + cohesive_dissipation 
        if returnall:
            return czm_functional, strain_energy, ce_term1, ce_term2, ce_term3, cohesive_dissipation
        else:
            return czm_functional
        
    def assemble_jac_czm_functional(self, d, lambda_, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """       
       
        d_ce_term3 = self.get_cohesive_energy_lagrange_derivative(lambda_,d)
        d_cohesive_dissipation = self.get_cohesive_dissipation_derivative(d)
        
        jac_czm_functional = - d_ce_term3 + d_cohesive_dissipation
        if returnall:
            return jac_czm_functional, d_ce_term3, d_cohesive_dissipation
        else:
            return jac_czm_functional
        
    def assemble_lip_functional(self, D, u, returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain_lip(u)
       
        strain_energy = self.get_strain_energy(strain,D)
        
        bulk_dissipation = self.get_bulk_dissipation(D)
       
        lip_functional = strain_energy  + bulk_dissipation 
        if returnall:
            return lip_functional, strain_energy,  bulk_dissipation
        else:
            return lip_functional
        
    def assemble_jac_lip_functional(self, D, u, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """
        strain = self.get_strain_lip(u)
        d_strain_energy = self.get_strain_energy_derivative(strain, D)
        d_bulk_dissipation = self.get_bulk_dissipation_derivative(D)

        jac_clip_functional = (d_strain_energy + d_bulk_dissipation)
        if returnall:
            return jac_clip_functional, d_strain_energy, d_bulk_dissipation
        else:
            return jac_clip_functional

    def dissipation_act_bulk_coh(self,strain_str, stress_fun_str, jump_fun_str):
        
        step_elem_strain = np.array(strain_str)
        step_stress = np.array(stress_fun_str)
       
        totalbulkdisp = 0.
        total_bulk_str = []
        sigm = (step_stress[1:] + step_stress[:-1])/2.
                    
        for ie in range(step_elem_strain.shape[1]):
            deps = step_elem_strain[1:,ie] - step_elem_strain[:-1,ie]
            bulkdispe = self.params.dx*np.sum(deps*sigm)       
            totalbulkdisp += bulkdispe
            total_bulk_str.append(bulkdispe)

        step_w = np.array(jump_fun_str)
        totalcohesivedisp = 0
        total_cohesive_str = []
        for ie in range(step_w.shape[1]):
            dstepw = step_w[1:, ie] - step_w[:-1, ie]
            cohesivedispe = np.sum(dstepw*sigm)
            totalcohesivedisp += cohesivedispe
            total_cohesive_str.append(cohesivedispe)
        
    
        return totalcohesivedisp,totalbulkdisp

    def get_potential_energy(self, u, D):
        strain = self.get_strain(u)

        pot = self.get_strain_energy(strain, D)

        return pot

    def get_cohesive_energy(self,d, jump, lambda_ ):
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(jump, lambda_, d)
        Ecoh = -ce_term1 + ce_term2 - ce_term3
        return Ecoh
class Solver:
    """
    Coordinates the solution of the equilibrium equations and damage evolution.
    """

    def __init__(self, model, simulation_parameters):
        self.bulk_damage = BulkDamage(simulation_parameters.lc, simulation_parameters.Dm, simulation_parameters.get_len_mat(simulation_parameters.x))
        self.equilibrium_solver = EquilibriumSolver(model, simulation_parameters)
        self.equilibrium_solver_lip = EquilibriumSolver_Lip(model, simulation_parameters)
        self.functional = Functional(model, simulation_parameters, self.bulk_damage)
        self.params = simulation_parameters
        
    def clip_czm_functional(self, d, bc):

        if self.params.functional_choice == 'CLIP-3terms':
            D_center = self.bulk_damage.get_Bulk_damage(d)
            u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc)        
            clip_functional = self.functional.assemble_clip_functional_3_terms(d, D_center, u_, lambda_)
            
        elif self.params.functional_choice == 'CLIP-4terms':
            D_center = self.bulk_damage.get_Bulk_damage(d)
            u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc) 
            clip_functional = self.functional.assemble_clip_functional_4_terms(d, D_center, u_, lambda_)
        
        elif self.params.functional_choice == 'CZM':
            D_pseudo_czm = (np.concatenate([np.ones(1), np.ones_like(d)]))
            u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_pseudo_czm, bc)      
            clip_functional = self.functional.assemble_czm_functional(d, D_pseudo_czm, u_, lambda_)
        
        else:
           raise ValueError("Invalid functional choice")

        return clip_functional
  
    def jac_functional(self, d, bc): 

        if self.params.functional_choice == 'CLIP-3terms': 
            D_center = self.bulk_damage.get_Bulk_damage(d)
            u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc)
            jac_clip_functional = self.functional.assemble_jac_clip_functional_3_terms(d, D_center, u_, lambda_)
                
        elif self.params.functional_choice == 'CLIP-4terms':
            D_center = self.bulk_damage.get_Bulk_damage(d)
            u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc)
            jac_clip_functional = self.functional.assemble_jac_clip_functional_4_terms(d, D_center, u_, lambda_)

        elif self.params.functional_choice == 'CZM':
            D_pseudo_czm =(np.concatenate([np.ones(1), np.ones_like(d)]))
            u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_pseudo_czm, bc)            
            jac_clip_functional = self.functional.assemble_jac_czm_functional(d, lambda_)
        
        else :
            raise ValueError("Invalid functional choice")

        return jac_clip_functional

    def clip_czm_functional_plot(self, d, bc):

            if self.params.functional_choice == 'CLIP-3terms':
                D_center = self.bulk_damage.get_Bulk_damage(d)
                u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc)        
                clip_functional = self.functional.assemble_clip_functional_3_terms(d, D_center, u_, lambda_)
                
            elif self.params.functional_choice == 'CLIP-4terms':
                D_center = self.bulk_damage.get_Bulk_damage(d)
                u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc) 
                functional_1, se, ce1, ce2, ce3, coh_dissip, bulk_dissip = self.functional.assemble_clip_functional_4_terms(d, D_center, u_, lambda_, returnall=True)
            
            elif self.params.functional_choice == 'CZM':
                D_pseudo_czm = (np.concatenate([np.ones(1), np.ones_like(d)]))
                u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_pseudo_czm, bc)      
                clip_functional = self.functional.assemble_czm_functional(d, D_pseudo_czm, u_, lambda_)
            
            else:
                raise ValueError("Invalid functional choice")

            return functional_1, se, ce1, ce2, ce3, coh_dissip, bulk_dissip ,lambda_
        
    def solve_functional(self, d, d_prev, bc):
        """
        Solves the functional for the given damage state and boundary conditions.
        """
        d1 = np.zeros(self.params.N_elements-1)
        dtest = np.linspace(0,1,1000)
        func_str = []
        se_str = []
        ce1_str = []
        ce2_str = []
        ce3_str = []
        coh_dissip_str = []
        bulk_dissip_str = []
        for i in range(1000):

                d1[math.floor((self.params.N_elements-1)/2)] = dtest[i]
                functional_1, se, ce1, ce2, ce3, coh_dissip, bulk_dissip,lda = self.clip_czm_functional_plot(d1, bc)
                func_str.append(functional_1)
                se_str.append(se)
                ce1_str.append(ce1)
                ce2_str.append(ce2)
                ce3_str.append(ce3)
                coh_dissip_str.append(coh_dissip)
                bulk_dissip_str.append(bulk_dissip)
                
        print("min func = ", min(func_str))
        #print("lda = ", lda, lda[math.floor((self.params.N_elements-1)/2)])
        plt.plot(dtest, func_str, label = ' Functional')
        # plt.plot(dtest, se_str, label = ' Strain Energy')
        # plt.plot(dtest, ce1_str, label = ' Cohesive Energy - 1')
        # plt.plot(dtest, ce2_str, label = ' Cohesive Energy - 2')
        # plt.plot(dtest, ce3_str, label = ' Cohesive Energy - 3')
        # plt.plot(dtest, coh_dissip_str, label = ' Cohesive Dissipation')
        # plt.plot(dtest, bulk_dissip_str, label = ' Bulk Dissipation')
        plt.xlabel('Cohesive Damage', fontsize = 'large')
        plt.ylabel('Functional (N/m)', fontsize = 'large')
        plt.title('Functional (F) vs Cohesive damage (d)', fontweight = 'bold', fontsize = 'large')
        
        plt.legend()
        plt.grid(True)
        plt.show()

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=dtest, y=func_str, mode='lines', name='Functional'))
        # fig.add_trace(go.Scatter(x=dtest, y=se_str, mode='lines', name='Strain_Energy'))
        # fig.add_trace(go.Scatter(x=dtest, y=ce1_str, mode='lines', name='Coh energy 1'))
        # fig.add_trace(go.Scatter(x=dtest, y=ce2_str, mode='lines', name='Coh energy 2'))
        # fig.add_trace(go.Scatter(x=dtest, y=ce3_str, mode='lines', name='Coh energy 3'))
        # fig.add_trace(go.Scatter(x=dtest, y=coh_dissip_str, mode='lines', name='Cohesive_Dissipation'))
        # fig.add_trace(go.Scatter(x=dtest, y=bulk_dissip_str, mode='lines', name='Bulk Dissipation'))
        
        # fig.update_layout(
        # xaxis=dict(showgrid=True, gridcolor='black',linecolor = 'black'),
        # yaxis=dict(showgrid=True, gridcolor='black', linecolor = 'black'),
        # title=f"Functional terms vs Cohesive Damage (d)",  # Using f-string
        # xaxis_title="Cohesive Damage",
        # yaxis_title=" Functional term (N/m)",
        # )
        # fig.show()

        functional = lambda damage : self.clip_czm_functional(damage, bc)
        jacobian = lambda damage : self.jac_functional(damage, bc)
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))

        damage_opt = scipy.optimize.minimize(
            fun = functional,
            jac = jacobian,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
        )
        if not damage_opt.success:
           raise RuntimeError("Optimization failed")
        return damage_opt
    
    def  lip_functional(self, D, u_, low_bound):

        functional = lambda D:self.functional.assemble_lip_functional(D,u_)
        jacob = lambda D: self.functional.assemble_jac_lip_functional(D,u_)
        A = scipy.sparse.eye(self.params.N_v-2,self.params.N_v-1) - scipy.sparse.eye(self.params.N_v-2,self.params.N_v-1,1)           
        slopeconstrain = scipy.optimize.LinearConstraint(A, -self.params.dx/self.params.lc *np.ones(self.params.N_v-2), self.params.dx/self.params.lc*np.ones(self.params.N_v-2) )
        bounds = scipy.optimize.Bounds(low_bound,np.ones(len(D)))

        damage_predictor_opt = scipy.optimize.minimize(
            fun=functional,
            x0=D,
            bounds=bounds ,            
            method = 'SLSQP',         
            constraints=slopeconstrain,
            jac =jacob
            )
                
        if not damage_predictor_opt.success :
            raise RuntimeError("Optimization failed")
        return damage_predictor_opt
    
class ExplicitSolver:

    def __init__(self,functions,parameters):
        self.bulk_damage = BulkDamage(parameters.lc, parameters.Dm,parameters.get_len_mat(parameters.x))
        self.functions = functions
        self.params = parameters
        self.Solver = Solver
        self.functional = Functional(functions, parameters, self.bulk_damage)
        self.w_max = 0
        self.d_max = 0
    
    def get_nodes(self):
        if self.params.new_crack == 0 :
            nodes = np.linspace(0, self.params.L, self.params.N_nodes)
            return nodes
        
        else : 
            nodes = np.concatenate((np.linspace(0, self.params.L/2, int((self.params.N_nodes+1)//2)),np.linspace(self.params.L/2, self.params.L, int((self.params.N_nodes+1)//2))))
            return nodes

    def initial_and_boundary_conditions(self, nodes):
        if self.params.boundary_type == 'free':
            vel = nodes * self.params.eps0dot
            return vel
    
        elif self.params.boundary_type == 'imposed' or self.params.boundary_type == 'small_case':
            vel = nodes * self.params.eps0dot
            return vel
        
        elif self.params.boundary_type == 'string':
            self.params.update_new_crack(2)
            d = np.zeros(self.params.N_elements-1)
            lda = np.zeros(self.params.N_elements-1)
            disp = nodes * (self.params.sigc * self.params.L)/(self.params.E) 
            mid_node = math.floor(self.params.N_nodes / 2)
            disp = np.insert(disp, mid_node, disp[mid_node])
            d[math.floor((self.params.N_elements-1)/2)] = 0.01
            stress, strain, _ = self.get_stress(disp, d, lda)
            force = self.get_nodal_forces(stress)
            mass = self.get_M_lumped()
            acc = self.compute_acceleration(force, mass)
            vel_predict = np.zeros(self.params.N_nodes-1)
            return disp, acc, vel_predict, d

    def boundary_conditions_on_time(self, time, disp):
        if self.params.boundary_type == 'free':
            return disp 
        
        elif self.params.boundary_type == 'imposed':
            disp[0] = 0
            disp[-1]  = self.params.L * self.params.eps0dot * time
            return disp 
  
    def velocity_predict(self, dt, vel, acc):
        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string':
            return vel[1:] + (dt /2)* acc[1:]
        elif self.params.boundary_type == 'imposed':
            return vel[1:-1] + (dt /2)* acc[1:-1]
        elif self.params.boundary_type == 'small_case':
            return vel + (dt /2)* acc

    def compute_displacement(self, dt, disp, vel, acc, vel_predict):
        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string': 
            #disp[1:] = disp[1:] + (dt*vel[1:]) + ((dt**2)/2)*acc[1:]
            disp[1:] = disp[1:] + (dt * vel_predict)
            return disp
        elif self.params.boundary_type == 'imposed':     
            disp[1: -1] += (dt * vel_predict)
            return disp
        elif self.params.boundary_type == 'small_case':
            disp += (dt * vel_predict)
            return disp
        
    def  compute_lagrange(self, disp, lda, stress):
        mid_index = math.floor(len(lda)/2)
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
        
        if self.params.new_crack == 0:
            return lda
        
        if self.params.new_crack == 1:
            lda[mid_index] = (stress[int(self.params.N_elements/2)-1] + 
                              stress[int(self.params.N_elements/2)])/2
            return lda
        
        else:
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half] )
            
            lda[mid_index]  = (self.params.sigc *(1-jump/self.params.wc)) + self.params.k*jump
            
            return lda
    
    def values_w_t(self):
    
        d_test = np.linspace(0, 1, 100000)
        D_test = self.params.Dm * d_test

        k = self.params.k
        yc = self.params.yc
        Yc = self.params.Yc
        lc = self.params.lc
        E = self.params.E

        gd = self.functions.gd_cohesive.get_value(d_test)
        g1d = self.functions.gd_cohesive.get_first_derivative(d_test)
        h1d =  self.functions.hd_cohesive.get_first_derivative(d_test)
        GDmd = self.functions.GD_bulk.get_value(D_test)
        HDmd = self.functions.HD_bulk.get_value(D_test)

        t_test = np.sqrt((((yc*h1d)/2) + ((Yc*lc*HDmd)))/(((lc)/(2*E))*(1/GDmd -1) - ((g1d)/(4*k*gd**2))))
        w_test = t_test/(k*gd)
        
        return w_test, t_test, d_test

    def traction_predict(self, w_known, opt = True):

        if w_known >= self.w_max and w_known <= self.params.wc:
            self.w_max = w_known

            w_test, t_test, d_test = self.values_w_t()
            interp_function = interp1d(w_test, t_test, kind = 'linear',fill_value = 'extrapolate')
            interp_damage_function = interp1d(w_test, d_test, kind = 'linear', fill_value = 'extrapolate')
            t_predicted = interp_function(w_known)
            d_predicted = interp_damage_function(w_known)
            self.d_max = d_predicted

        elif w_known > self.params.wc:            
            t_predicted = 0
            d_predicted = 1
        
        elif w_known < 0 :
            t_predicted = 0
            d_predicted = self.d_max
            w_known = 0
        
        else :
            t_predicted = self.params.k * self.functions.gd_cohesive.get_value(self.d_max) * w_known
            d_predicted = self.d_max

        if opt:
            return t_predicted
        else:
            return t_predicted, d_predicted, w_known

    def compute_lda_clip(self, disp, lda, stress):
        mid_index = math.floor(len(lda)/2)
        N_nodes_half = math.ceil(self.params.N_nodes / 2)

        if self.params.new_crack == 0:
            return lda

        if self.params.new_crack == 1 :
            lda[mid_index] = (stress[int(self.params.N_elements/2)-1] + stress[int(self.params.N_elements/2)])/2            
            return lda
        
        else:
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            lda[mid_index] = self.traction_predict(jump) + self.params.k*jump            
            return lda

    def compute_damage(self, d, d_prev, lda):
        
        def F(d):
            lda_lda_on_k = (lda * lda)/self.params.k
            return -0.5*np.dot(lda_lda_on_k,(self.functions.gd_cohesive.get_lmb_value(d))) + self.params.yc*np.sum(self.functions.hd_cohesive.get_value(d))

        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))
        damage_opt = scipy.optimize.minimize(
            fun = F,
            #jac = jacobian,
            x0 = d,
            bounds = bounds,
            #method = 'SLSQP'
        )
        if not damage_opt.success:
           raise RuntimeError("Optimization failed")
        return damage_opt.x

    def compute_damage_clip(self, d, d_prev, disp, lda):
    
        d1 = np.zeros(self.params.N_elements-1)
        dtest = np.linspace(0,1,1000)
        func_str = []
        se_str = []
        ce1_str = []
        ce2_str = []
        ce3_str = []
        coh_dissip_str = []
        bulk_dissip_str = []
        for i in range(1000):

                d1[math.floor((self.params.N_elements-1)/2)] = dtest[i]
                D_center = self.bulk_damage.get_Bulk_damage(d1)
                functional_1, se, ce1, ce2, ce3, coh_dissip, bulk_dissip = self.functional.assemble_clip_functional_4_terms_explicit_clip(d1, D_center, disp, lda, returnall = True)
                func_str.append(functional_1)
                se_str.append(se)
                ce1_str.append(ce1)
                ce2_str.append(ce2)
                ce3_str.append(ce3)
                coh_dissip_str.append(coh_dissip)
                bulk_dissip_str.append(bulk_dissip)
                
        print("min func = ", min(func_str))
        #print("lda = ", lda, lda[math.floor((self.params.N_elements-1)/2)])
        plt.plot(dtest, func_str, label = ' Functional')
        # plt.plot(dtest, se_str, label = ' Strain Energy')
        # plt.plot(dtest, ce1_str, label = ' Cohesive Energy - 1')
        # plt.plot(dtest, ce2_str, label = ' Cohesive Energy - 2')
        # plt.plot(dtest, ce3_str, label = ' Cohesive Energy - 3')
        # plt.plot(dtest, coh_dissip_str, label = ' Cohesive Dissipation')
        # plt.plot(dtest, bulk_dissip_str, label = ' Bulk Dissipation')
        plt.xlabel('Cohesive Damage', fontsize = 'large')
        plt.ylabel('Functional (N/m)', fontsize = 'large')
        plt.title('Functional (F) vs Cohesive damage (d)', fontweight = 'bold', fontsize = 'large')
        
        plt.legend()
        plt.grid(True)
        plt.show()

        D_center = self.bulk_damage.get_Bulk_damage(d)
        functional = lambda damage : self.functional.assemble_clip_functional_4_terms_explicit_clip(damage,D_center,disp, lda)
        
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))
        damage_opt = scipy.optimize.minimize(
            fun = functional,
            #jac = jacobian,
            x0 = d,
            bounds = bounds,
            #method = 'SLSQP'
        )
        if not damage_opt.success:
           raise RuntimeError("Optimization failed")
        return damage_opt.x

    def get_strain(self, disp):
        if self.params.new_crack == 0:
            return (disp[1:] - disp[:-1])/self.params.dx
        
        else:
            # For new crack, split the displacement into two parts (left and right)
            mid_node = math.ceil(self.params.N_nodes / 2)
            
            disp_left = disp[:mid_node]
            disp_right = disp[mid_node:]

            # Calculate strains for the left and right segments of the displacement array
            strain_left = (disp_left[1:] - disp_left[:-1]) / self.params.dx
            strain_right = (disp_right[1:] - disp_right[:-1]) / self.params.dx

            # Combine strains from both segments
            return np.concatenate((strain_left, strain_right))

    def get_stress(self, disp, d, lda):

        N_nodes_half = math.ceil(self.params.N_nodes / 2)
        lda_mid = math.floor(len(lda) / 2)

        if self.params.new_crack == 0 :
            strain = self.get_strain(disp)     
            sig = self.params.E * strain 
            return sig, strain, sig
        
        else:
            D_center = self.bulk_damage.get_Bulk_damage(d)
            disp_left = disp[: N_nodes_half]
            disp_right = disp[N_nodes_half: ]

            strain_left = (disp_left[1:] - disp_left[:-1])/self.params.dx
            strain_right = (disp_right[1:] - disp_right[:-1])/self.params.dx
            
            D_left = D_center[: N_nodes_half-1]
            D_right = D_center[N_nodes_half-1:]
            
            sig_left = (self.params.E*strain_left* self.functions.GD_bulk.get_value(D_left))
            sig_right = (self.params.E*strain_right*self.functions.GD_bulk.get_value(D_right))

            sig_left_GD = (self.params.E*strain_left)
            sig_right_GD = (self.params.E*strain_right)

            if self.params.new_crack == 1 :       
                traction = self.params.sigc
                self.params.update_new_crack(2)
                
            else:       
                jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
                traction = self.params.k*(self.functions.gd_cohesive.get_value(d[lda_mid]))*jump
            
            sig_new = np.concatenate((sig_left, [traction], sig_right))
            sig_bulk = np.concatenate((sig_left, sig_right))
            return sig_new , np.concatenate((strain_left, strain_right)), sig_bulk

    def get_nodal_forces(self, stress):
        if self.params.new_crack == 0:
            nodal_forces = np.zeros(self.params.N_nodes)
            nodal_forces[0:self.params.N_nodes-1] += self.params.Area*stress
            nodal_forces[1:self.params.N_nodes] -= self.params.Area*stress
            return nodal_forces
        else :
            nodal_forces = np.zeros(self.params.N_nodes+1)
            nodal_forces[0:self.params.N_nodes] += self.params.Area*stress
            nodal_forces[1:self.params.N_nodes+1] -= self.params.Area*stress
            return nodal_forces

    def get_M_lumped(self):

        element_mass = self.params.Area * self.params.rho * self.params.dx
        half_mass = element_mass / 2
        N_nodes = self.params.N_nodes
        M = np.zeros(N_nodes + (1 if self.params.new_crack else 0))

        M[1:-1] = element_mass
        M[0] = half_mass
        M[-1] = half_mass

        if self.params.new_crack != 0 :
            mid_index = math.ceil(N_nodes / 2) 
            M[mid_index - 1] = half_mass
            M[mid_index] = half_mass

        return M

    def compute_acceleration(self, force, mass):
        return np.divide(force,mass, where=mass!=0)

    def compute_velocity(self, dt, vel, vel_predict, acc):
        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string':
            vel[1:] =  vel_predict + (dt/2)*acc[1:]
            return vel
        
        elif self.params.boundary_type == 'imposed':
            vel[1:-1] =  vel_predict + (dt/2)*acc[1:-1]
            return vel
            
        elif self.params.boundary_type == 'small_case':
            vel =  vel_predict + (dt/2)*acc
            return vel
    def checkcohesivestress(self, disp, vel, acc, stress):
        mid_elem = int(self.params.N_elements / 2)
        Stress_avg = (stress[mid_elem-1] + stress[mid_elem])/2
        #Stress_avg = (stress[0] + stress[1])/2
        
        if Stress_avg > self.params.sigc and self.params.new_crack == 0 :   
                self.params.update_new_crack(1)
                mid_node = math.floor(self.params.N_nodes / 2)

                disp = np.insert(disp, mid_node, disp[mid_node])
                vel = np.insert(vel, mid_node, vel[mid_node])
                acc = np.insert(acc, mid_node, acc[mid_node])

                sigc = Stress_avg
                self.params.update_sigc(sigc)
        
        return disp, vel, acc

    def init_Energy_compute(self, disp, vel, lda, d, dt):
        Area = self.params.Area
        dx = self.params.dx
        E = self.params.E

        Epot, Ekin, Edissip = 0.0, 0.0, 0.0
        ext_work = 0

        strain = self.get_strain(disp)     
        Epot = 0.5 * Area * dx * E *(np.dot(strain,strain))

        M = self.get_M_lumped()
        Ekin = 0.5 * np.dot((M* vel), vel)

        Edissip =  self.params.yc * np.sum(self.functions.hd_cohesive.get_value(d)) * Area
        
        stress = self.get_stress(disp, d, lda)
        ext_work = (stress[-1] * vel[-1] * Area* dt)

        init_energy = {
        "pot":Epot,
        "kin": Ekin,
        "dissip" : Edissip,
        "ext": ext_work,
        }

        return init_energy

    def Energy_computation(self, disp, vel, lda, d, stress, dt, Ext_work,  init_energy = None):

        Area = self.params.Area
        dx = self.params.dx
        E = self.params.E
        Gc = self.params.Gc
        N_nodes_half = math.ceil(self.params.N_nodes / 2) 
        Epot, Ekin, Edissip, Ecoh = 0.0, 0.0, 0.0, 0.0
        Edissip_bulk, Edissip_coh = 0.0, 0.0

        D_nodes, D, _ = self.bulk_damage.get_Bulk_damage(d, centeronly=False)
 
        Ext_work += -((stress[-1]*vel[-1]*Area*dt) + (stress[0]*vel[0]*Area*dt))
        #Ext_work += -(stress[-1]*disp[-1]*Area)

        if self.params.new_crack == 0 :
            strain = self.get_strain(disp)
           
            Epot = 0.5 * Area * dx * E *(np.dot((strain * strain),self.functions.GD_bulk.get_value(D)) )

        else :
            disp_left = disp[: N_nodes_half]
            disp_right = disp[N_nodes_half : ]
            
            strain_left = (disp_left[1:] - disp_left[:-1])/dx
            strain_right = (disp_right[1:] - disp_right[:-1])/dx
            E_pot_left = 0.5 * Area  * dx * E *   (np.dot((strain_left *strain_left), self.functions.GD_bulk.get_value(D[:N_nodes_half-1])))
            E_pot_right = 0.5 * Area  * dx * E * (np.dot((strain_right*strain_right), self.functions.GD_bulk.get_value(D[N_nodes_half-1:])))

            Epot = E_pot_left + E_pot_right

        M = self.get_M_lumped()
        Ekin = 0.5 * np.dot((M* vel), vel)

        if self.params.new_crack !=0:
            jump = np.zeros(self.params.N_elements-1)
            jump[math.floor(self.params.N_elements / 2)-1] =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            
            gd =  np.where(np.isinf(self.functions.gd_cohesive.get_value(d)), 0, self.functions.gd_cohesive.get_value(d))
            
            Ecoh = (0.5 * self.params.k * np.sum(gd * np.sum(jump**2)))*Area

        if self.params.new_crack != 0:
            Edissip_coh =  self.params.yc * np.sum(self.functions.hd_cohesive.get_value(d)) * Area
            Edissip_bulk = self.params.Yc * np.sum(self.functions.HD_bulk.get_value(D_nodes)) * Area * dx
            
            Edissip = Edissip_bulk + Edissip_coh

        Epot_var = Epot 
        Ekin_var = Ekin 
        Edissip_var = Edissip 
        Eext_var = Ext_work  

        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string' or self.params.boundary_type == 'small_case':
            Total_energy = Epot_var + Ekin_var  + Edissip_var + Ecoh
            
        elif self.params.boundary_type == 'imposed':
            Total_energy = Epot_var + Ekin_var  + Edissip_var + Ecoh + Ext_work


            
        return Epot_var, Ekin_var, Edissip_var, Eext_var, Ecoh, Total_energy, Edissip_bulk, Edissip_coh
