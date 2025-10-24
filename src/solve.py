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
        self.K_uu_ju = None
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

        nlines = data.shape[0]
        # Create or update the K_ll matrix
        if self.K_ll is None:
            # Generate indices for creating a square matrix
            indices = np.arange(nlines)
            # Construct a sparse matrix with damage-derived data
            self.K_ll = scipy.sparse.coo_matrix((data, (indices, indices)), shape=(nlines, nlines))
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

    def get_K_uu_ju(self,d):
        """
        Constructs or updates the joint stiffness matrix.
        """
        # Check if the joint stiffness matrix already exists
        if self.K_uu_ju  is None:
            # Define the pattern for stiffness contributions between nodes
            stiffness_pattern = np.array([[ self.params.k, -self.params.k, -self.params.k, self.params.k]])
            d = np.where(d == 0, 0.000000000001, d)
            element_stiffness_contributions = (self.model.gd_cohesive.get_value(d)[:, np.newaxis] * stiffness_pattern).flatten()
            
            # Generate column and row indices for COO matrix format
            col_indices = np.repeat(np.arange(1, (2 * self.params.N_elements - 1)).reshape((-1, 2)), 2, axis=0).flatten()
            row_indices = np.repeat(np.arange(1, (2 * self.params.N_elements) - 1), 2)
            
            # Create the sparse COO format joint stiffness matrix with the defined patterns
            self.K_uu_ju = scipy.sparse.coo_matrix((element_stiffness_contributions, (row_indices, col_indices)),
                                                shape=(2 * self.params.N_elements, 2 * self.params.N_elements))
        else :
            stiffness_pattern = np.array([[ self.params.k, -self.params.k, -self.params.k, self.params.k]])
            d = np.where(d == 0, 0.00000000001, d)
            element_stiffness_contributions = (self.model.gd_cohesive.get_value(d)[:, np.newaxis] * stiffness_pattern).flatten()
            self.K_uu_ju.data = element_stiffness_contributions

        return self.K_uu_ju

    def get_K_u(self, d, D_center, imposed_displacements):
        
       # Retrieve or update the bulk and joint stiffness matrices based on current damage and displacement
        K_uu_b = self.get_K_uu_b(D_center)
        K_uu_j = self.get_K_uu_ju(d)
        
        # Update or retrieve matrices for coupling and boundary conditions
        K_bc = self.get_K_bc(list(imposed_displacements.keys()))

        # If the global stiffness matrix K already exists, update its data
        if self.K is None:

            # Assemble data for the new global stiffness matrix
            data = np.hstack([K_uu_b.data, K_uu_j.data,  K_bc.data, K_bc.data])

            row_indices = np.hstack([K_uu_b.row, K_uu_j.row,  K_bc.row, 
                                     K_bc.col + 2*self.params.N_elements])
            col_indices = np.hstack([K_uu_b.col, K_uu_j.col, K_bc.col + 2*self.params.N_elements, 
                                     K_bc.row])

            size = 2 * self.params.N_elements  + len(imposed_displacements.keys())
            
            # Create the global stiffness matrix
            self.K = scipy.sparse.coo_matrix((data, (row_indices, col_indices)), shape=(size, size))
    
        else:
            self.K.data[:self.K_uu_b.nnz] = K_uu_b.data
            self.K.data[K_uu_b.nnz:K_uu_b.nnz + K_uu_j.nnz] = K_uu_j.data
        return self.K   

    def F_vector(self, imposed_displacements):

        # Determine the size of the force vector if it has not been initialized
        if self.F is None:
            self.F = np.zeros(2 * self.params.N_elements + len(imposed_displacements))
        
        # Update the portion of the force vector corresponding to imposed displacements
        # The last segment of the vector is set based on the values of the imposed displacements
        end_index = 2 * self.params.N_elements
        imposed_values = np.array(list(imposed_displacements.values()))
        self.F[end_index:] = imposed_values
        
        return self.F

    def solve_equilibrium_u(self, d, D_center, imposed_displacements):

        # Assemble the global stiffness matrix for the current state
        K = self.get_K_u(d, D_center, imposed_displacements)
        
        # Construct the global force vector based on imposed displacements
        F = self.F_vector(imposed_displacements)
        
        # Solve the linear system to find displacements and Lagrange multipliers
        u = scipy.sparse.linalg.spsolve(K, F)

        # Extract displacements and Lagrange multipliers from the solution vector
        # Displacements at nodes
        resx = u[:(2 * self.params.N_elements)]
       
        # Lagrange multipliers for boundaries (enforcing boundary conditions)
        resL_b = -u[(2 * self.params.N_elements):]
        
        return resx, resL_b, K

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
        # return (u[1:] - u[:-1])/ self.params.dx
    
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
        d = np.where(d==0, 0.000000000001, d)
        # gd = np.where(np.isinf(self.model.gd_cohesive.get_value(d)), 0, self.model.gd_cohesive.get_value(d))
        # # gd =  np.where(d < 1e-8, 0, self.model.gd_cohesive.get_value(d))
        return 0.5 * self.params.k * (self.model.gd_cohesive.get_value(d)).dot(u_jump**2)
        
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

    def assemble_clip_functional_4_terms(self, d, u, lambda_,returnall = False):
        """
        Assemble the functional to minimize
        """
 
        strain = self.get_strain(u)
        u_jump = self.get_jump(u)
    
        D_center = self.bulk_damage.get_Bulk_damage(d)

        strain_energy = self.get_strain_energy(strain, D_center)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)        
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)

        clip_functional = strain_energy - ce_term1 + ce_term2 - ce_term3 + bulk_dissipation + cohesive_dissipation 
        
        if returnall:
            return clip_functional, strain_energy, ce_term1, ce_term2, ce_term3, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional

    def assemble_clip_functional_gd_4_terms(self, d, u, returnall = False):
        """
        Assemble the functional to minimize
        """
        D_center = self.bulk_damage.get_Bulk_damage(d)
        strain = self.get_strain(u)
        u_jump = self.get_jump(u)

        strain_energy = self.get_strain_energy(strain, D_center)
        cohesive_energy = self.get_cohesive_energy(u_jump, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)
   
        clip_functional = strain_energy + cohesive_energy + cohesive_dissipation + bulk_dissipation
        if returnall:            
            return clip_functional, strain_energy, cohesive_energy, bulk_dissipation, cohesive_dissipation
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
     
    def assemble_clip_functional_4_terms_explicit_clip(self, d, u, lambda_, returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain_explicit_clip(u)
        u_jump = self.get_jump_explcit_clip(u)
        D_center = self.bulk_damage.get_Bulk_damage(d)
        
        strain_energy = self.get_strain_energy(strain, D_center)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)

        clip_functional = strain_energy - ce_term1 + ce_term2 - ce_term3 + bulk_dissipation + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, ce_term1, ce_term2, ce_term3, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional

    def assemble_clip_functional_4_terms_explicit_clip_gd(self, d, u,returnall = False):
        """
        Assemble the functional to minimize
        """
        D_center = self.bulk_damage.get_Bulk_damage(d)
        strain = self.get_strain_explicit_clip(u)
        u_jump = self.get_jump_explcit_clip(u)

        strain_energy = self.get_strain_energy(strain, D_center)
        d = np.where(d==0, 0.00000001, d)
        coh_energy = 0.5 * self.params.k * (self.model.gd_cohesive.get_value(d).dot(u_jump**2))
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)
      
        clip_functional = strain_energy + coh_energy + bulk_dissipation + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, coh_energy, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional

    def assemble_clip_functional_4_terms_explicit_clip_test(self, d, u,returnall = False):
        """
        Assemble the functional to minimize
        """
        D_center = self.bulk_damage.get_Bulk_damage(d)
        strain = self.get_strain_explicit_clip(u)
        u_jump = self.get_jump_explcit_clip(u)

        strain_energy = self.get_strain_energy(strain, D_center)
        d = np.where(d==0, 1 , d)
        coh_energy = 0.5 * self.params.k * (self.model.gd_cohesive.get_value(d).dot(u_jump**2))
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)
      
        clip_functional = strain_energy + coh_energy + bulk_dissipation + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, coh_energy, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional

    def assemble_clip_functional_4_terms_explicit_clip_lda(self, d, u, returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain_explicit_clip(u)
        u_jump = self.get_jump_explcit_clip(u)
        D_center = self.bulk_damage.get_Bulk_damage(d)
        d_lda = np.where(d==0, 0.0000000000000000001, d)
        lambda_ = ( 1 + self.model.gd_cohesive.get_value(d_lda))*self.params.k*u_jump
        
        strain_energy = self.get_strain_energy(strain, D_center)
        ce_term1, ce_term2, ce_term3 = self.get_cohesive_energy_lagrange(u_jump, lambda_, d)
        cohesive_dissipation = self.get_cohesive_dissipation(d)
        bulk_dissipation = self.get_bulk_dissipation(D_center)

        clip_functional = strain_energy - ce_term1 + ce_term2 - ce_term3 + bulk_dissipation + cohesive_dissipation 
        if returnall:
            return clip_functional, strain_energy, ce_term1, ce_term2, ce_term3, bulk_dissipation, cohesive_dissipation
        else:
            return clip_functional

    def assemble_jac_clip_functional_4_terms_gd(self, d, u, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """
        D_center = self.bulk_damage.get_Bulk_damage(d)
        strain = self.get_strain(u)
        u_jump = self.get_jump(u)
        
        d_strain_energy = self.get_strain_energy_derivative(strain, D_center)
        d_ce_term = self.get_cohesive_energy_derivative(u_jump, d)
        d_bulk_dissipation = self.get_bulk_dissipation_derivative(D_center)
        d_cohesive_dissipation = self.get_cohesive_dissipation_derivative(d)
        dD_center = self.bulk_damage.get_dBulk_damage_dd(d)

        jac_clip_functional = dD_center.T.dot(d_strain_energy + d_bulk_dissipation) - d_ce_term + d_cohesive_dissipation
        if returnall:
            return jac_clip_functional, d_strain_energy, d_ce_term, d_bulk_dissipation, d_cohesive_dissipation
        else:
            return jac_clip_functional

    def assemble_jac_clip_functional_4_terms(self, d, u, lambda_, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """

        strain = self.get_strain(u)
        u_jump = self.get_jump(u)
        D_center = self.bulk_damage.get_Bulk_damage(d)

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

    def dissipation_act_bulk_coh_lip(self,strain_str, stress_fun_str):
        
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

    
        return totalbulkdisp

    def get_potential_energy(self, u, D):
        strain = self.get_strain(u)

        pot = self.get_strain_energy(strain, D)

        return pot

    def get_cohesive_energy_lda(self,d, jump, lambda_ ):
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
            clip_functional = self.functional.assemble_clip_functional_4_terms(d, u_, lambda_)
        
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
            jac_clip_functional = self.functional.assemble_jac_clip_functional_4_terms(d, u_, lambda_)

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
        # d1 = np.zeros(self.params.N_elements-1)
        # dtest = np.linspace(0,1,1000)
        # func_str = []
        # se_str = []
        # ce1_str = []
        # ce2_str = []
        # ce3_str = []
        # coh_dissip_str = []
        # bulk_dissip_str = []
        # for i in range(1000):

        #         d1[math.floor((self.params.N_elements-1)/2)] = dtest[i]
        #         functional_1, se, ce1, ce2, ce3, coh_dissip, bulk_dissip,lda = self.clip_czm_functional_plot(d1, bc)
        #         func_str.append(functional_1)
        #         se_str.append(se)
        #         ce1_str.append(ce1)
        #         ce2_str.append(ce2)
        #         ce3_str.append(ce3)
        #         coh_dissip_str.append(coh_dissip)
        #         bulk_dissip_str.append(bulk_dissip)
                
        # print("min func = ", min(func_str))
        # #print("lda = ", lda, lda[math.floor((self.params.N_elements-1)/2)])
        # plt.plot(dtest, func_str, label = ' Functional')
        # # plt.plot(dtest, se_str, label = ' Strain Energy')
        # # plt.plot(dtest, ce1_str, label = ' Cohesive Energy - 1')
        # # plt.plot(dtest, ce2_str, label = ' Cohesive Energy - 2')
        # # plt.plot(dtest, ce3_str, label = ' Cohesive Energy - 3')
        # # plt.plot(dtest, coh_dissip_str, label = ' Cohesive Dissipation')
        # # plt.plot(dtest, bulk_dissip_str, label = ' Bulk Dissipation')
        # plt.xlabel('Cohesive Damage', fontsize = 'large')
        # plt.ylabel('Functional (N/m)', fontsize = 'large')
        # plt.title('Functional (F) vs Cohesive damage (d)', fontweight = 'bold', fontsize = 'large')
        
        # plt.legend()
        # plt.grid(True)
        # plt.show()

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

    def solve_gd_functional(self,d, d_prev,  u_):
        """
        Solves the functional for the given damage state and boundary conditions.
        """
        
        functional = lambda damage : self.functional.assemble_clip_functional_gd_4_terms(damage, u_)
        jacobian = lambda damage : self.functional.assemble_jac_clip_functional_4_terms_gd( damage, u_)
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))

        damage_opt = scipy.optimize.minimize(
            fun = functional,
            # jac = jacobian,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
        )
        # if not damage_opt.success:
        #    raise RuntimeError("Optimization failed")
        return damage_opt

    def solve_lda_functional(self,d, d_prev, u_, lambda_):
        """
        Solves the functional for the given damage state and boundary conditions.
        """
        
        functional = lambda damage : self.functional.assemble_clip_functional_4_terms(damage, u_, lambda_)
        # jacobian = lambda damage : self.functional.assemble_jac_clip_functional_4_terms( damage, D, u_, lambda_)
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))

        damage_opt = scipy.optimize.minimize(
            fun = functional,
            # jac = jacobian,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
        )
        # if not damage_opt.success:
        #    raise RuntimeError("Optimization failed")
        return damage_opt

    def solve_alt_ulda_d(self, d, u_, lambda_, bc, trigermiddamage = False, i = 1):
        # iteration = 0
        # d=d_0.copy() 
        # doutputi  = int((self.params.N_elements - 1)/2)
        # energy_old = self.functional.assemble_clip_functional_4_terms(d, u_, lambda_) 
        # while (iteration < 1000) :
        #     d_0 = d.copy()
        #     if trigermiddamage :
        #         d_0[doutputi] = 0.01 
        #     D = self.bulk_damage.get_Bulk_damage(d,centeronly = True)
        #     u_, _, lambda_, _ = self.equilibrium_solver.solve_equilibrium_ul(d, D, bc)
        #     results = self.solve_lda_functional(d, d_0, u_, lambda_)
        #     d = results.x
        #     energy = self.functional.assemble_clip_functional_4_terms(d, u_, lambda_) 
        #     print('d solved , energy =', energy)
        #     print("d = ", d[doutputi ])
        #     iteration += 1
        #     abs_norm_delta_fun = np.abs((energy-energy_old)) 
        #     if abs_norm_delta_fun < 1.e-8: 
        #         break
        #     if  abs_norm_delta_fun/energy < 1.e-8:
        #         break 
            
        #     energy_old = energy
        #     # norm_delta_fun = np.linalg.norm(solver.functional.assemble_clip_functional_4_terms(d, u_, lambda_) - solver.functional.assemble_clip_functional_4_terms(d_prev , u_prev, lmb_prev))/np.linalg.norm(solver.functional.assemble_clip_functional_4_terms(d, u_, lambda_))
        #     print("it = ",iteration,"nrm_fun = ",abs_norm_delta_fun)

        iteration = 0
        stop = False
        d_0 = d.copy()
        D = np.zeros(self.params.N_elements)
        doutputi  = int((self.params.N_elements - 1)/2)
        while iteration < 1000 and not stop:
            u_prev = u_.copy()
            print(u_prev)
            
            lmb_prev = lambda_.copy()

            if trigermiddamage :
                d_prev = np.zeros(self.params.N_elements - 1)
                d_prev[doutputi] = 0.1
            else:
                d_prev = d.copy()
            
            energy_old = self.functional.assemble_clip_functional_4_terms(d_prev , u_prev, lmb_prev)
            _, D, D_overall = self.bulk_damage.get_Bulk_damage(d,centeronly = False)

            u_, F_fun, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D, bc)
        
            results = self.solve_lda_functional(d, d_prev, u_, lambda_)
            d = results.x
            
            print("d = ", d[doutputi])
            
            iteration += 1

            energy = self.functional.assemble_clip_functional_4_terms(d, u_, lambda_)

            norm_delta_fun = np.linalg.norm(energy - energy_old )/np.linalg.norm(energy)

            if norm_delta_fun < 1e-8  or i == 0:
                stop = True
            print("it = ",iteration,"nrm_fun = ",norm_delta_fun)
        return u_, lambda_, d, D

    def functional_plot(self, d, d_prev, disp, lambda_):
        d1 = np.zeros(self.params.N_elements-1)
        dtest = np.linspace(0.000,1,100)
        func_str = []
        se_str = []
        ce1_str = []
        ce2_str = []
        ce3_str = []
        coh_dissip_str = []
        bulk_dissip_str = []
        for i in range(100):

                d1[math.floor((self.params.N_elements-1)/2)] = dtest[i]
                
                D_center = self.bulk_damage.get_Bulk_damage(d1)
                functional_1, se, ce1, ce2, ce3, bulk_dissip, coh_dissip = self.functional.assemble_clip_functional_4_terms(d1, D_center, disp, lambda_, returnall = True)
                func_str.append(functional_1)
                se_str.append(se)
                ce1_str.append(ce1)
                ce2_str.append(ce2)
                ce3_str.append(ce3)
                coh_dissip_str.append(coh_dissip)
                bulk_dissip_str.append(bulk_dissip)
                
        print("min func = ", min(func_str))
        # #print("lda = ", lda, lda[math.floor((self.params.N_elements-1)/2)])
        # #plt.plot(dtest, func_str, label = ' Functional')
        # # plt.plot(dtest, se_str, label = ' Strain Energy')
        # # plt.plot(dtest, ce1_str, label = ' Cohesive Energy - 1')
        # # plt.plot(dtest, coh_dissip_str, label = ' Cohesive Dissipation')
        # # plt.plot(dtest, bulk_dissip_str, label = ' Bulk Dissipation')
        # plt.xlabel('Cohesive Damage', fontsize = 'large')
        # plt.ylabel('Functional (N/m)', fontsize = 'large')
        # plt.title('Functional (F) vs Cohesive damage (d)', fontweight = 'bold', fontsize = 'large')
        
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dtest, y=func_str, mode='lines', name='Functional'))
        fig.add_trace(go.Scatter(x=dtest, y=se_str, mode='lines', name='Strain Energy'))
        fig.add_trace(go.Scatter(x=dtest, y=ce1_str, mode='lines', name='Cohesive Energy - 1'))
        fig.add_trace(go.Scatter(x=dtest, y=ce2_str, mode='lines', name='Cohesive Energy - 2'))
        fig.add_trace(go.Scatter(x=dtest, y=ce3_str, mode='lines', name='Cohesive Energy - 3'))
        fig.add_trace(go.Scatter(x=dtest, y=coh_dissip_str, mode='lines', name='Cohesive Dissipation'))
        fig.add_trace(go.Scatter(x=dtest, y=bulk_dissip_str, mode='lines', name='Bulk Dissipation'))
        
        fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor='black',linecolor = 'black'),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor = 'black'),
        title=f"Functional terms vs Cohesive Damage (d)",  # Using f-string
        xaxis_title="Cohesive Damage",
        yaxis_title=" Functional term (N/m)",
        )
        fig.write_html("functional_Dm_09.html")
        fig.show()
        
        print("")

    def functional_gd_plot(self, d, d_prev, disp, lambda_):
        d1 = np.zeros(self.params.N_elements-1)
        dtest = np.linspace(0.000,1,100)
        func_str = []
        se_str = []
        ce1_str = []
        ce2_str = []
        ce3_str = []
        coh_dissip_str = []
        bulk_dissip_str = []
        for i in range(100):

                d1[math.floor((self.params.N_elements-1)/2)] = dtest[i]
                
                D_center = self.bulk_damage.get_Bulk_damage(d1)
                functional_1, se, ce,  bulk_dissip,coh_dissip = self.functional.assemble_clip_functional_gd_4_terms(d1, D_center, disp, lambda_, returnall = True)
                func_str.append(functional_1)
                se_str.append(se)
                ce1_str.append(ce)
                coh_dissip_str.append(coh_dissip)
                bulk_dissip_str.append(bulk_dissip)
                
        print("min func = ", min(func_str))
        # #print("lda = ", lda, lda[math.floor((self.params.N_elements-1)/2)])
        # #plt.plot(dtest, func_str, label = ' Functional')
        # # plt.plot(dtest, se_str, label = ' Strain Energy')
        # # plt.plot(dtest, ce1_str, label = ' Cohesive Energy - 1')
        # # plt.plot(dtest, coh_dissip_str, label = ' Cohesive Dissipation')
        # # plt.plot(dtest, bulk_dissip_str, label = ' Bulk Dissipation')
        # plt.xlabel('Cohesive Damage', fontsize = 'large')
        # plt.ylabel('Functional (N/m)', fontsize = 'large')
        # plt.title('Functional (F) vs Cohesive damage (d)', fontweight = 'bold', fontsize = 'large')
        
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dtest, y=func_str, mode='lines', name='Functional'))
        fig.add_trace(go.Scatter(x=dtest, y=se_str, mode='lines', name='Strain Energy'))
        fig.add_trace(go.Scatter(x=dtest, y=ce1_str, mode='lines', name='Cohesive Energy'))
        fig.add_trace(go.Scatter(x=dtest, y=coh_dissip_str, mode='lines', name='Cohesive Dissipation'))
        fig.add_trace(go.Scatter(x=dtest, y=bulk_dissip_str, mode='lines', name='Bulk Dissipation'))
        
        fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor='black',linecolor = 'black'),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor = 'black'),
        title=f"Functional terms vs Cohesive Damage (d)",  # Using f-string
        xaxis_title="Cohesive Damage",
        yaxis_title=" Functional term (N/m)",
        )
        fig.write_html("functional_Dm_09.html")
        fig.show()
        
        print("")

    def lip_functional(self, D, u_, low_bound):

        functional = lambda D:self.functional.assemble_lip_functional(D,u_)
        jacob = lambda D: self.functional.assemble_jac_lip_functional(D,u_)
        
        A = scipy.sparse.eye(self.params.N_v-2,self.params.N_v-1) - scipy.sparse.eye(self.params.N_v-2,self.params.N_v-1,1)           
        # A=scipy.sparse.eye(self.params.N_nodes-1,self.params.N_nodes) - scipy.sparse.eye(self.params.N_nodes-1,self.params.N_nodes,1)         
        slopeconstrain = scipy.optimize.LinearConstraint(A, -self.params.dx/self.params.lc *np.ones(self.params.N_v-2), self.params.dx/self.params.lc*np.ones(self.params.N_v-2) )
        # slopeconstrain = scipy.optimize.LinearConstraint(A, -self.params.dx/self.params.lc *np.ones(self.params.N_nodes-1), self.params.dx/self.params.lc*np.ones(self.params.N_nodes-1) )
        bounds = scipy.optimize.Bounds(low_bound,np.ones(len(D)))
        # print('Damage =', D)
        # print("disp =", u_)
 
        damage_predictor_opt = scipy.optimize.minimize(
            fun=functional,
            x0=D,
            bounds=bounds ,            
            method = 'SLSQP',         
            constraints=slopeconstrain,
            # jac =jacob
            )
                
        if not damage_predictor_opt.success :
            raise RuntimeError("Optimization failed")
        return damage_predictor_opt
    
class ExplicitSolver:

    def __init__(self,functions,parameters):
        self.bulk_damage = BulkDamage(parameters.lc, parameters.Dm,parameters.get_len_mat(parameters.x))
        self.functions = functions
        self.params = parameters
        self.Solver = Solver(functions, parameters)
        self.Equib_solver = EquilibriumSolver(functions, parameters)
        self.functional = Functional(functions, parameters, self.bulk_damage)
        self.w_max = 0
        self.bulk_dissipation = 0.0 
        self.cohesive_dissipation = 0.0 
        self.d_max = 0
        self.prev_jump = 0.0
        self.previous_strain = 0.0
        self.previous_stress = 0.0
        self.prev_traction = 0.0
        self.test = 0.0

    def call_method(self,D,disp, D_prev):
        return self.Solver.lip_functional(self,D,disp, D_prev)

    def get_nodes(self):
        if self.params.new_crack == 0 :
            nodes = np.linspace(0, self.params.L, self.params.N_nodes)
            return nodes
        
        else : 
            nodes = np.concatenate((np.linspace(0, self.params.L/2, int((self.params.N_nodes+1)//2)),np.linspace(self.params.L/2, self.params.L, int((self.params.N_nodes+1)//2))))
            return nodes

    def initial_and_boundary_conditions(self, nodes):
        if self.params.boundary_type == 'free':
            disp = np.zeros(self.params.N_nodes)
            acc = np.zeros(self.params.N_nodes)
            d = np.zeros(self.params.N_elements-1)
            vel_predict = np.zeros(self.params.N_nodes-1)
            vel = nodes * self.params.eps0dot 
            return disp, acc, vel, vel_predict, d
    
        elif self.params.boundary_type == 'imposed' or self.params.boundary_type == 'small_case':
            vel = nodes * self.params.eps0dot
            return vel
        
        elif self.params.boundary_type == 'string':
            self.params.update_new_crack(2)
            d = np.zeros(self.params.N_elements-1)
            lda = np.zeros(self.params.N_elements-1)
            disp = nodes * (self.params.sigc )/(self.params.E)
            mid_node = math.floor(self.params.N_nodes / 2)
            disp = np.insert(disp, mid_node, disp[mid_node])
            d[math.floor((self.params.N_elements-1)/2)] = 0.001
            
            stress, strain, _, traction = self.get_stress(disp, d, lda)
            w_test, t_test, d_test = self.values_w_t()
            interp_jump_function = interp1d(d_test, w_test, kind = 'linear', fill_value = 'extrapolate')
            self.w_max = interp_jump_function(d[math.floor((self.params.N_elements-1)/2)])
        
            # self.w_max = self.params.sigc / (self.params.k * self.functions.gd_cohesive.get_value(d[math.floor((self.params.N_elements-1)/2)]))

            force = self.get_nodal_forces(stress)
            mass = self.get_M_lumped()
            acc = self.compute_acceleration(force, mass)
            vel = np.zeros(self.params.N_nodes+1)
            vel_predict = np.zeros(self.params.N_nodes - 1)

            acc[0] = 0
            acc[-1] = 0
            return disp, acc, vel, vel_predict, d
    
    def apply_boundary_conditions(self,acc):

        if self.params.boundary_type =='string':
            acc[0] = 0
            acc[-1] = 0
            return acc
        
        elif self.params.boundary_type == 'free':
            acc[0] = 0
            return acc

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

    def compute_displacement(self, dt, disp, vel_predict):

        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string':

            disp[1:] = disp[1:] + (dt * vel_predict)
            return disp

        elif self.params.boundary_type == 'imposed':

            disp[1: -1] += (dt * vel_predict)
            return disp
        
    def compute_lagrange(self, disp, lda, stress):
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
    
        d_test = np.linspace(0., 1, 100000)
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
        
        #t_test = np.sqrt((((yc * h1d)) + (Yc * lc * HDmd))/(((lc)/(2 * E) * (1/GDmd -1)) - ((g1d)/(2*k*gd**2))))

        w_test = t_test/(k*gd)
        
        return w_test, t_test, d_test

    def traction_predict(self, w_known,d_mid, opt = True):
        
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
            t_predicted = self.params.k * self.functions.gd_cohesive.get_value(d_mid) * w_known
            d_predicted = d_mid

        if opt:
            return t_predicted
        else:
            return t_predicted, d_predicted, w_known

    def compute_lda_clip(self, disp, lda, d, stress):
        mid_index = math.floor(len(lda)/2)
        N_nodes_half = math.ceil(self.params.N_nodes / 2)

        if self.params.new_crack == 1 :
            print("lda 1")
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            print("Jump = ", jump)
            lda[mid_index] = (stress[int(self.params.N_elements/2)-1] + stress[int(self.params.N_elements/2)])/2
            self.test = 1
            return lda
        elif self.params.new_crack == 2:
            print("lda 3")
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            print("Jump = ", jump)
            print("d_pred = ", self.traction_predict(jump, d[mid_index],opt = False))
            lda[mid_index] = (1 + self.functions.gd_cohesive.get_value(d[mid_index]))*self.params.k*jump
            return lda

    def compute_lda_clip_1(self, disp, lda, d, stress):
        mid_index = math.floor(len(lda)/2)
        N_nodes_half = math.ceil(self.params.N_nodes / 2)

        if self.params.new_crack == 1 :
            lda[mid_index] = (stress[int(self.params.N_elements/2)-1] + stress[int(self.params.N_elements/2)])/2
            return lda
        
        else:
            jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            print("Jump = ", jump, d[mid_index])
            # traction , d[mid_index], w_known= self.traction_predict(jump, d[mid_index],opt = False)
            # print("Traction = ", traction, d[mid_index], w_known)
            lda[mid_index] = stress + self.params.k*jump
            # lda[mid_index] = (1 + self.functions.gd_cohesive.get_value(d[mid_index]))*self.params.k*jump
            
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
    
    def compute_damage_clip_gd(self, d, d_prev, disp):

        functional = lambda damage : self.functional.assemble_clip_functional_4_terms_explicit_clip_gd(damage,disp)
        
        if d_prev[math.floor((self.params.N_elements-1)/2)] == 0.:
            d_prev[math.floor((self.params.N_elements-1)/2)] = 0.01

        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))
        damage_opt = scipy.optimize.minimize(
            fun = functional,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
        )
        # if not damage_opt.success:
        #    raise RuntimeError("Optimization failed")
        return damage_opt.x
    
    def compute_damage_clip_test(self, d, d_prev, disp):

        functional = lambda damage : self.functional.assemble_clip_functional_4_terms_explicit_clip_test(damage,disp)
        
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))
        damage_opt = scipy.optimize.minimize(
            fun = functional,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
        )
        # if not damage_opt.success:
        #    raise RuntimeError("Optimization failed")
        return damage_opt.x

    def compute_damage_clip(self, d, d_prev, disp, lda):
    
        # d1 = np.zeros(self.params.N_elements-1)
        # dtest = np.linspace(0,1,1000)
        # func_str = []
        # se_str = []
        # ce1_str = []
        # ce2_str = []
        # ce3_str = []
        # coh_dissip_str = []
        # bulk_dissip_str = []
        # for i in range(1000):

        #         d1[math.floor((self.params.N_elements-1)/2)] = dtest[i]
        #         D_center = self.bulk_damage.get_Bulk_damage(d1)
        #         functional_1, se, ce1, ce2, ce3, coh_dissip, bulk_dissip = self.functional.assemble_clip_functional_4_terms_explicit_clip(d1, D_center, disp, lda, returnall = True)
        #         func_str.append(functional_1)
        #         se_str.append(se)
        #         ce1_str.append(ce1)
        #         ce2_str.append(ce2)
        #         ce3_str.append(ce3)
        #         coh_dissip_str.append(coh_dissip)
        #         bulk_dissip_str.append(bulk_dissip)
                
        # print("min func = ", min(func_str))
        # #print("lda = ", lda, lda[math.floor((self.params.N_elements-1)/2)])
        # plt.plot(dtest, func_str, label = ' Functional')
        # # plt.plot(dtest, se_str, label = ' Strain Energy')
        # # plt.plot(dtest, ce1_str, label = ' Cohesive Energy - 1')
        # # plt.plot(dtest, ce2_str, label = ' Cohesive Energy - 2')
        # # plt.plot(dtest, ce3_str, label = ' Cohesive Energy - 3')
        # # plt.plot(dtest, coh_dissip_str, label = ' Cohesive Dissipation')
        # # plt.plot(dtest, bulk_dissip_str, label = ' Bulk Dissipation')
        # plt.xlabel('Cohesive Damage', fontsize = 'large')
        # plt.ylabel('Functional (N/m)', fontsize = 'large')
        # plt.title('Functional (F) vs Cohesive damage (d)', fontweight = 'bold', fontsize = 'large')
        
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        functional = lambda damage : self.functional.assemble_clip_functional_4_terms_explicit_clip(damage, disp, lda)
        jacobian = lambda damage : self.functional.assemble_jac_clip_functional_4_terms(damage, disp, lda)
        
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))
        damage_opt = scipy.optimize.minimize(
            fun = functional,
            # jac = jacobian,
            x0 = d,
            bounds = bounds,
            # method = 'SLSQP'
        )
        if not damage_opt.success:
           raise RuntimeError("Optimization failed")
        return damage_opt.x

    def compute_damage_combo(self, d, d_prev, disp, lda):
        """
        Simultaneously optimize the damage (d) and Lagrange multiplier (lambda, lda)
        by treating them as a combined optimization vector.
        
        Parameters:
        - d: Initial damage vector.
        - d_prev: Previous damage vector (lower bound for d).
        - disp: Displacement (fixed).
        - lda: Initial Lagrange multipliers.

        Returns:
        - Optimized damage (d_opt) and Lagrange multipliers (lda_opt).
        """
        # Combine d and lambda into a single vector for optimization
        x0 = np.concatenate([d, lda])

        # Define the objective function that combines damage and lambda
        def functional(x):
            # Split the combined vector into damage and lambda
            damage = x[:len(d)]
            lamb = x[len(d):]
            # Assemble the functional using the split values
            return self.functional.assemble_clip_functional_4_terms_explicit_clip(damage, disp, lamb)
        
        # Define bounds for d and lambda
        bounds_d = [(d_prev[i], 1.0) for i in range(len(d))]
        bounds_lda = [(None, None) for _ in lda]  # No bounds on lambda
        bounds = bounds_d + bounds_lda

        # Use a constrained optimization solver
        result = scipy.optimize.minimize(
            fun=functional,       # Objective function
            x0=x0,                # Initial combined vector
            bounds=bounds,        # Bounds for damage and lambda
            # method='SLSQP'        # Sequential Least Squares Programming
        )
        
        # Raise an error if optimization fails
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        # Split the optimized result back into damage and lambda
        d_opt = result.x[:len(d)]
        lda_opt = result.x[len(d):]
        
        return d_opt, lda_opt

    def compute_damage_lda_clip(self, d, d_prev, disp):
    
        functional = lambda damage : self.functional.assemble_clip_functional_4_terms_explicit_clip_lda(damage, disp)
        
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))
        damage_opt = scipy.optimize.minimize(
            fun = functional,
            #jac = jacobian,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
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
            return sig, strain, sig, 0.
        
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
                print("traction =", traction)
                
            else:       
                jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
                traction = self.params.k*(self.functions.gd_cohesive.get_value(d[lda_mid]))*jump
                print("traction =", traction, jump)

            sig_new = np.concatenate((sig_left, [traction], sig_right))
            sig_bulk = np.concatenate((sig_left, sig_right))
            return sig_new , np.concatenate((strain_left, strain_right)), sig_bulk, traction

    def get_lda_stress(self, disp, d, lda):

        N_nodes_half = math.ceil(self.params.N_nodes / 2)
        lda_mid = math.floor(len(lda) / 2)

        if self.params.new_crack == 0 :
            strain = self.get_strain(disp)     
            sig = self.params.E * strain 
            return sig, strain, sig, 0.
        
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
                traction = lda[lda_mid]
                self.params.update_new_crack(2)
                
            else:       
                jump =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
                traction = lda[lda_mid] - (self.params.k*jump)
                print("traction = ", traction, jump)
                # traction = self.params.k*jump*self.functions.gd_cohesive.get_value(d[lda_mid])
            
            sig_new = np.concatenate((sig_left, [traction], sig_right))
            sig_bulk = np.concatenate((sig_left, sig_right))
            return sig_new , np.concatenate((strain_left, strain_right)), sig_bulk, traction

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
        print("mass = ", element_mass)
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

        elif self.params.boundary_type == 'lip':
            vel[1:] =  vel_predict + (dt/2)*acc[1:]
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
                print("Cohesive zone inserted, Avg Stress =",Stress_avg, self.params.sigc)
        
        return disp, vel, acc

    def checkcohesivestress_test(self, disp, vel, acc, stress, lda):

        mid_elem = int(self.params.N_elements / 2)
        Stress_avg = (stress[mid_elem-1] + stress[mid_elem])/2
        
        if Stress_avg > self.params.sigc and self.params.new_crack == 0 :   
                self.params.update_new_crack(1)
                mid_node = math.floor(self.params.N_nodes / 2)

                disp = np.insert(disp, mid_node, disp[mid_node])
                vel = np.insert(vel, mid_node, vel[mid_node])
                acc = np.insert(acc, mid_node, acc[mid_node])
                lda[mid_elem-1] = Stress_avg
                # sigc = Stress_avg
                # self.params.update_sigc(sigc)
        
        return disp, vel, acc, lda

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

    def Energy_computation(self, disp, vel, lda, d, stress, strain, dt, Ext_work):

        Area = self.params.Area
        dx = self.params.dx
        E = self.params.E
        k = self.params.k
        Gc = self.params.Gc
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
    
        total_energy = {'Epot': 0.0, 'Ekin': 0.0, 'Edissip': 0.0, 'Edissip_bulk': 0.0, 'Edissip_coh': 0.0, 'Ecoh': 0.0, 'Ext_work': Ext_work, 'Total': 0.0}
        
        D_nodes, D, _ = self.bulk_damage.get_Bulk_damage(d, centeronly=False)

        # Potential Energy
        total_energy['Epot'] = 0.5 * Area * dx * E *(np.dot((strain * strain),self.functions.GD_bulk.get_value(D)) )

        # Kinetic Energy
        M = self.get_M_lumped()
        total_energy['Ekin'] = 0.5 * np.dot((M * vel), vel)

        #Cohesive Energy
        if self.params.new_crack !=0:
            jump = np.zeros(self.params.N_elements-1)
            jump[math.floor(self.params.N_elements / 2)-1] =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            d = np.where(d==0, 0.0000000001, d)
            # gd =  np.where(np.isinf(self.functions.gd_cohesive.get_value(d)), 0, self.functions.gd_cohesive.get_value(d))
            # gd =  np.where(d < 1e-6, 0, self.functions.gd_cohesive.get_value(d))
            total_energy['Ecoh'] = (0.5 * self.params.k * np.sum(self.functions.gd_cohesive.get_value(d[math.floor(self.params.N_elements / 2)-1]) * np.sum(jump[math.floor(self.params.N_elements / 2)-1]**2)))*Area

        # Dissipative Energy
        if self.params.new_crack != 0:
            total_energy['Edissip_coh'] =  self.params.yc * np.sum(self.functions.hd_cohesive.get_value(d)) * Area
            total_energy['Edissip_bulk'] = self.params.Yc * np.sum(self.functions.HD_bulk.get_value(D_nodes)) * Area * dx
            total_energy['Edissip'] = total_energy['Edissip_bulk'] + total_energy['Edissip_coh']

        # External Work
        total_energy['Ext_work'] += -((stress[-1]*vel[-1]*Area*dt) + (stress[0]*vel[0]*Area*dt))

        # Total Energy
        if self.params.boundary_type in ['free', 'string', 'small_case']:
            total_energy['Total'] = total_energy['Epot'] + total_energy['Ekin'] + total_energy['Ecoh'] + total_energy['Edissip']
            
        elif self.params.boundary_type == 'imposed':
            total_energy['Total'] = total_energy['Epot'] + total_energy['Ekin'] + total_energy['Ecoh'] + total_energy['Edissip'] + total_energy['Ext_work']

        return total_energy

    def Energy_verlet_computation(self, disp, vel, vel_predict, d, stress,traction, strain, dt, Ext_work):

        Area = self.params.Area
        dx = self.params.dx
        E = self.params.E
        k = self.params.k
        Gc = self.params.Gc
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
    
        total_energy_verlet = {'Epot': 0.0, 'Ekin': 0.0, 'Edissip': 0.0, 'Edissip_bulk': 0.0, 'Edissip_coh': 0.0, 'Ecoh': 0.0, 'Ext_work': Ext_work, 'Total': 0.0}
        
        D_nodes, D, _ = self.bulk_damage.get_Bulk_damage(d, centeronly=False)


        total_energy_verlet['Epot'] = 0.5 * Area * dx * E *(np.dot((strain * self.previous_strain),self.functions.GD_bulk.get_value(D)))
        strain_increment = strain - self.previous_strain
        self.previous_strain = strain

        if self.test == 0 :
            self.test = 1
            total_energy_verlet['Epot'] = 0.5 * Area * dx * E *(np.dot((strain * strain),self.functions.GD_bulk.get_value(D)) )

        stress_average = (stress + self.previous_stress )/2
        bulk_dissipation_increment = dx * Area * np.dot(stress_average , strain_increment)
        self.bulk_dissipation += bulk_dissipation_increment
        #total_energy_verlet['Edissip_bulk'] = (self.bulk_dissipation - total_energy_verlet['Epot'])
        total_energy_verlet['Edissip_bulk'] = (self.bulk_dissipation -  0.5 * Area * dx * E *(np.dot((strain * strain),self.functions.GD_bulk.get_value(D))))
        self.previous_stress = stress

        # Kinetic Energy
        M = self.get_M_lumped()
        total_energy_verlet['Ekin'] = 0.5 * np.dot((M[1:] * vel_predict), vel_predict)

        #Cohesive Energy
        if self.params.new_crack !=0:
            jump = np.zeros(self.params.N_elements-1)
            jump[math.floor(self.params.N_elements / 2)-1] =  abs(disp[N_nodes_half-1] - disp[N_nodes_half])
            # d = np.where(d == 0, 0.0000000001, d)
            total_energy_verlet['Ecoh'] = (0.5 * self.params.k * np.sum(self.functions.gd_cohesive.get_value(d[math.floor(self.params.N_elements / 2)-1]) * np.sum(jump[math.floor(self.params.N_elements / 2)-1]*self.prev_jump)))*Area

            traction_average = (traction + self.prev_traction )/2
            self.cohesive_dissipation += np.sum(traction_average * (jump - self.prev_jump))* Area 
            total_energy_verlet['Edissip_coh'] = (self.cohesive_dissipation - (0.5 * self.params.k * np.sum(self.functions.gd_cohesive.get_value(d[math.floor(self.params.N_elements / 2)-1]) * np.sum(jump[math.floor(self.params.N_elements / 2)-1]*jump[math.floor(self.params.N_elements / 2)-1])))*Area )
            #total_energy_verlet['Edissip_coh'] = (self.cohesive_dissipation - total_energy_verlet['Ecoh'])
            self.prev_traction = traction
            self.prev_jump = jump


        total_energy_verlet['Edissip'] = total_energy_verlet['Edissip_bulk'] + total_energy_verlet['Edissip_coh']

        # External Work
        total_energy_verlet['Ext_work'] += -((stress[-1]*vel[-1]*Area*dt) + (stress[0]*vel[0]*Area*dt))

        # Total Energy
        if self.params.boundary_type in ['free', 'string', 'small_case']:
            total_energy_verlet['Total'] = total_energy_verlet['Epot'] + total_energy_verlet['Ekin'] + total_energy_verlet['Ecoh'] + total_energy_verlet['Edissip']
            
        elif self.params.boundary_type == 'imposed':
            total_energy_verlet['Total'] = total_energy_verlet['Epot'] + total_energy_verlet['Ekin'] + total_energy_verlet['Ecoh'] + total_energy_verlet['Edissip'] + total_energy_verlet['Ext_work']

        return total_energy_verlet, strain_increment

    def disp_vel_predict_alpha(self,dt, disp, vel, acc, alpha_m, alpha_f):
        disp_pred = disp[1:] + (dt * vel[1:]) + (0.5*dt**2)*(1-2*alpha_f)*acc[1:]
        vel_pred = vel[1:] + dt*(1-alpha_m)*acc[1:]
        return disp_pred, vel_pred

    def disp_vel_compute_alpha(self, dt, disp_pred, vel_pred, acc, alpha_f, alpha_m):
        disp_comp = disp_pred[1:] + alpha_f*dt**2*acc[1:]
        vel_comp = vel_pred + dt*(alpha_m)*acc[1:]
        return disp_comp, vel_comp
        
    def quasi_stat_exp_dynamic(self, disp, lda, d):
        disp_mid = math.floor(len(disp) / 2)
        d_center = d[disp_mid:disp_mid+1]
        
        bc1 = disp[disp_mid-2]
        bc2 = disp[disp_mid+1]
        bc = {0: bc1, 3: bc2}
        stop = False
        iteration = 0
        d_prev = d_center.copy()
        while iteration < 100 and not stop:
            D_center = np.array([(d_center - (self.params.dx)/(2*self.params.lc))*self.params.Dm  ,(d_center - (self.params.dx)/(2*self.params.lc))*self.params.Dm])
    
            disp_test, _ , lda_center, _ = self.Equib_solver.solve_equilibrium_ul(d_center, D_center, bc)
            
            result = self.Solver.solve_lda_functional(d_center, d_prev, disp_test, lda_center)
            d = result.x

            iteration += 1

            norm_delta_fun = np.linalg.norm(self.functional.assemble_clip_functional_4_terms(d, u_, lambda_) - self.functional.assemble_clip_functional_4_terms(d_prev , u_prev, lmb_prev))/np.linalg.norm(self.functional.assemble_clip_functional_4_terms(d, u_, lambda_))

            if norm_delta_fun < 1e-5:
                stop = True    
    

        return disp, d

class ExplicitSolverLip:

    def __init__(self,functions,parameters):
        self.bulk_damage = BulkDamage(parameters.lc, parameters.Dm,parameters.get_len_mat(parameters.x))
        self.functions = functions
        self.params = parameters
        self.Solver = Solver
        self.functional = Functional(functions, parameters, self.bulk_damage)
        self.w_max = 0
        self.bulk_dissipation = 0.0 
        self.cohesive_dissipation = 0.0 
        self.d_max = 0
        self.prev_jump = 0.0
        self.previous_strain = 0.0
        self.previous_stress = 0.0
        self.prev_traction = 0.0

    def get_nodes(self):
            nodes = np.linspace(0, self.params.L, self.params.N_nodes)
            return nodes
      
    def initial_and_boundary_conditions(self, nodes):
        if self.params.boundary_type == 'free':
            disp = np.zeros(self.params.N_nodes)
            acc = np.zeros(self.params.N_nodes)
            d = np.zeros(self.params.N_elements-1)
            vel_predict = np.zeros(self.params.N_nodes-1)
            vel = nodes * self.params.eps0dot
            return disp, acc, vel, vel_predict, d
    
        elif self.params.boundary_type == 'string':
            disp = nodes * (self.params.sigc * self.params.L)/(self.params.E)
            vel = np.zeros(self.params.N_nodes)
            D = np.zeros(self.params.N_nodes)
            D[math.floor((self.params.N_nodes-1)/2)] = 0.6
            
            vel_predict = np.zeros(self.params.N_nodes-1)
            D_center = self.get_Bulk_damage_center(D)
            stress, strain = self.get_stress(disp, D_center)

            force = self.get_nodal_forces(stress)
            mass = self.get_M_lumped()
            acc = self.compute_acceleration(force, mass)

            return disp, acc, vel, vel_predict, D

    def velocity_predict(self, dt, vel, acc):
        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string':
            return vel + (dt /2)* acc

    def compute_displacement(self, dt, disp, vel_predict):

        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string':

            disp = disp + (dt * vel_predict)
            return disp

    def call_method(self,D,disp, D_prev):
        return self.Solver.lip_functional(self,D,disp, D_prev)

    def get_Bulk_damage_center(self, D):
        D_center = (D[1:] + D[:-1])/2
        return D_center

    def get_stress(self, disp, D_center):

        strain = (disp[1:] - disp[:-1])/self.params.dx     
        sig = self.params.E * strain * self.functions.GD_bulk.get_value(D_center)
        return sig, strain

    def get_nodal_forces(self, stress):
 
        nodal_forces = np.zeros(self.params.N_nodes)
        nodal_forces[0:self.params.N_nodes-1] += self.params.Area*stress
        nodal_forces[1:self.params.N_nodes] -= self.params.Area*stress
        return nodal_forces

    def get_M_lumped(self):

        element_mass = self.params.Area * self.params.rho * self.params.dx
        half_mass = element_mass / 2
        N_nodes = self.params.N_nodes
        M = np.zeros(N_nodes)

        M[1:-1] = element_mass
        M[0] = half_mass
        M[-1] = half_mass

        return M
    
    def compute_acceleration(self, force, mass):
        return np.divide(force,mass, where=mass!=0)

    def compute_velocity(self, dt, vel, vel_predict, acc):
        if self.params.boundary_type == 'free' or self.params.boundary_type == 'string':
            vel =  vel_predict + (dt/2)*acc
            return vel

    def Energy_computation(self, disp, vel, lda, D, stress, strain, dt, Ext_work):

        Area = self.params.Area
        dx = self.params.dx
        E = self.params.E
        k = self.params.k
        Gc = self.params.Gc
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
    
        total_energy = {'Epot': 0.0, 'Ekin': 0.0, 'Edissip': 0.0, 'Edissip_bulk': 0.0, 'Edissip_coh': 0.0, 'Ecoh': 0.0, 'Ext_work': Ext_work, 'Total': 0.0}
        
        # Potential Energy
        total_energy['Epot'] = 0.5 * Area * dx * E *(np.dot((strain * strain),self.functions.GD_bulk.get_value(D)) )

        # Kinetic Energy
        M = self.get_M_lumped()
        total_energy['Ekin'] = 0.5 * np.dot((M * vel), vel)

        # Dissipative Energy
        if self.params.new_crack != 0:
            total_energy['Edissip_bulk'] = self.params.Yc * np.sum(self.functions.HD_bulk.get_value(D_nodes)) * Area * dx
            total_energy['Edissip'] = total_energy['Edissip_bulk']

        # External Work
        total_energy['Ext_work'] += -((stress[-1]*vel[-1]*Area*dt) + (stress[0]*vel[0]*Area*dt))

        # Total Energy
        if self.params.boundary_type in ['free', 'string', 'small_case']:
            total_energy['Total'] = total_energy['Epot'] + total_energy['Ekin']  + total_energy['Edissip']
            
        return total_energy

    def Energy_verlet_computation(self, disp, vel, vel_predict, D_center, stress, strain, dt, Ext_work):

        Area = self.params.Area
        dx = self.params.dx
        E = self.params.E
        k = self.params.k
        Gc = self.params.Gc
        N_nodes_half = math.ceil(self.params.N_nodes / 2)
    
        total_energy_verlet = {'Epot': 0.0, 'Ekin': 0.0, 'Edissip': 0.0, 'Edissip_bulk': 0.0, 'Edissip_coh': 0.0, 'Ecoh': 0.0, 'Ext_work': Ext_work, 'Total': 0.0}
    
        # Potential Energy
        total_energy_verlet['Epot'] = 0.5 * Area * dx * E *(np.dot((strain * self.previous_strain),self.functions.GD_bulk.get_value(D_center)) )
        strain_increment = strain - self.previous_strain
        self.previous_strain = strain

        stress_average = (stress + self.previous_stress )/2
        bulk_dissipation_increment = dx * Area * np.dot(stress_average , strain_increment)
        self.bulk_dissipation += bulk_dissipation_increment
        #total_energy_verlet['Edissip_bulk'] = (self.bulk_dissipation - total_energy_verlet['Epot'])        total_energy_verlet['Edissip_bulk'] = (self.bulk_dissipation -  0.5 * Area * dx * E *(np.dot((strain * strain),self.functions.GD_bulk.get_value(D))))
        self.previous_stress = stress

        # Kinetic Energy
        M = self.get_M_lumped()
        total_energy_verlet['Ekin'] = 0.5 * np.dot((M * vel_predict), vel_predict)

        total_energy_verlet['Edissip'] = total_energy_verlet['Edissip_bulk'] 

        # External Work
        total_energy_verlet['Ext_work'] += -((stress[-1]*vel[-1]*Area*dt) + (stress[0]*vel[0]*Area*dt))

        # Total Energy
        if self.params.boundary_type in ['free', 'string', 'small_case']:
            total_energy_verlet['Total'] = total_energy_verlet['Epot'] + total_energy_verlet['Ekin']  + total_energy_verlet['Edissip']
            
        return total_energy_verlet
