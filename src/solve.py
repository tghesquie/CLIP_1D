import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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
        element_stiffness_contributions = (self.model.GD_bulk(D_center)[:, np.newaxis] * (self.params.E / self.params.dx * np.array([1., -1., -1., 1.]))).flatten()
        
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
       
        data = -d / self.params.k
        
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



class ClipFunctional:

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
        return 0.5 * self.params.dx * self.params.E * (self.model.GD_bulk(D_center).dot(strain**2))
    
    
    def get_strain_energy_derivative(self, strain, D_center):
        """
        Calculates the derivative of strain energy with respect to strain.
        """
        return 0.5 * self.params.dx * self.params.E * (self.model.dGD_bulk_dD(D_center) * (strain**2))
    

    def get_cohesive_energy(self, u_jump, d):
        """
        Calculates the cohesive energy for the given jump in displacement and damage state.
        """
        return 0.5 * self.params.k * (self.model.gd_cohesive(d).dot(u_jump**2))
        
    
    def get_cohesive_energy_derivative(self, u_jump, d):
        """
        Calculates the derivative of cohesive energy with respect to jump in displacement.
        """
        return 0.5 * self.params.k * (self.model.dgd_cohesive_dd(d) * (u_jump**2))
    

    def get_cohesive_dissipation(self, d):
        """
        Calculates the cohesive energy dissipation for the given damage state.
        """
        return np.sum(self.params.yc * self.model.hd_cohesive(d))
    
    
    def get_cohesive_dissipation_derivative(self, d):
        """
        Calculates the derivative of cohesive energy dissipation with respect to damage.
        """
        return self.params.yc * self.model.dhd_cohesive_dd(d)
    
    
    def get_bulk_dissipation(self, D_center):
        """
        Calculates the bulk energy dissipation for the given damage state.
        """
        return np.sum(self.params.Yc * self.params.dx * self.model.HD_bulk(D_center))
    

    def get_bulk_dissipation_derivative(self, D_center):
        """
        Calculates the derivative of bulk energy dissipation with respect to damage.
        """
        return self.params.Yc * self.params.dx * self.model.dHD_bulk_dD(D_center)
    

    def get_cohesive_energy_lagrange(self, u_jump, lambda_, d):
        """
        Calculates the cohesive energy when using Lagrange multipliers in the formulation.
        """
        term1 = 0.5 * self.params.k * np.sum(u_jump**2)
        term2 = np.dot(lambda_, u_jump)
        term3 = 1/(2 * self.params.k) * d.dot(lambda_**2)
        return term1, term2, term3
    

    def get_cohesive_energy_lagrange_derivative(self, lambda_):
        return 1/(2 * self.params.k) * lambda_**2

    
    def assemble_clip_functional(self, d, D_center, u, lambda_, returnall = False):
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
        

    def assemble_jac_clip_functional(self, d, D_center, u, lambda_, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """
        strain = self.get_strain(u)
        
        d_strain_energy = self.get_strain_energy_derivative(strain, D_center)
        d_ce_term3 = self.get_cohesive_energy_lagrange_derivative(lambda_)
        d_bulk_dissipation = self.get_bulk_dissipation_derivative(D_center)
        d_cohesive_dissipation = self.get_cohesive_dissipation_derivative(d)
        dD_center = self.bulk_damage.get_dBulk_damage_dd(d)

        jac_clip_functional = dD_center.T.dot(d_strain_energy + d_bulk_dissipation) - d_ce_term3 + d_cohesive_dissipation
        if returnall:
            return jac_clip_functional, d_strain_energy, d_ce_term3, d_bulk_dissipation, d_cohesive_dissipation
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


class Solver:
    """
    Coordinates the solution of the equilibrium equations and damage evolution.
    """

    def __init__(self, model, simulation_parameters):
        self.bulk_damage = BulkDamage(simulation_parameters.lc, simulation_parameters.Dm, simulation_parameters.get_len_mat(simulation_parameters.x))
        self.equilibrium_solver = EquilibriumSolver(model, simulation_parameters)
        self.functional = ClipFunctional(model, simulation_parameters, self.bulk_damage)
        
        


    def clip_functional(self, d, bc):
        D_center = self.bulk_damage.get_Bulk_damage(d)
        u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc)

        clip_functional = self.functional.assemble_clip_functional(d, D_center, u_, lambda_)
        return clip_functional
    

    def jac_clip_functional(self, d, bc):
        D_center = self.bulk_damage.get_Bulk_damage(d)
        u_, F, lambda_, K = self.equilibrium_solver.solve_equilibrium_ul(d, D_center, bc)

        jac_clip_functional = self.functional.assemble_jac_clip_functional(d, D_center, u_, lambda_)
        return jac_clip_functional

    
    def solve_functional(self, d, d_prev, bc):
        """
        Solves the functional for the given damage state and boundary conditions.
        """
        
        functional = lambda damage : self.clip_functional(damage, bc)
        jacobian = lambda damage : self.jac_clip_functional(damage, bc)
        bounds = scipy.optimize.Bounds(d_prev, np.ones(len(d_prev)))

        damage_opt = scipy.optimize.minimize(
            fun = functional,
            jac = jacobian,
            x0 = d,
            bounds = bounds,
            method = 'SLSQP'
        )
        if not damage_opt.success:
            print("Optimization failed")
        return damage_opt