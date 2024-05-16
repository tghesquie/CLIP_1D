import numpy as np
import scipy.sparse
import scipy.sparse.linalg

################################################################

class EquilibriumSolver:
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
    
    def B_matrix(self,):
        return -1./self.params.dx*scipy.sparse.eye(self.params.N_v-1,self.params.N_v) +1./self.params.dx*scipy.sparse.eye(self.params.N_v-1,self.params.N_v,1)
    
    def d2e_eps_deps2(self, d) : 
        return self.params.dx*scipy.sparse.diags(self.params.E*self.model.GD_bulk.get_Value(d))

    def K_uu(self,d):
        B = self.B_matrix()
        return B.T.dot(self.d2e_eps_deps2(d).dot(B))
    
    def solve_equilibrium_ul(self, d, imposed_displacements):
        Kuu = self.K_uu(d)
        n,n =Kuu.shape
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
        res = {}    
        res['x'] = u[:n]
        res['L'] = -u[n:]
        res['nit'] = 1
        return res['x'], res['L']
          
class LipFunctional:
    def __init__(self, model, simulation_parameters):
        """
        Initializes the LipFunctional with a model for material behaviors and simulation parameters.
        """
        self.model = model
        self.params = simulation_parameters
    
    def get_strain(self,u) :
        """
        Calculates the strain for a given displacement field.
        """
        return (u[1:] - u[:-1])/ self.params.dx
    
    def get_strain_energy(self,strain,D_center) :
        """
        Calculates the strain energy for the given strain and damage state.
        """
        return self.params.dx*0.5*(self.params.E*(self.model.GD_bulk.get_Value(D_center))*strain).dot(strain)

    def get_strain_energy_derivative(self, strain, D_center):
        """
        Calculates the derivative of strain energy with respect to damage.
        """
        return 0.5 * self.params.dx * self.params.E * (self.model.GD_bulk.get_First_derivative(D_center) * (strain**2))
    

    def get_bulk_dissipation(self, D_center):
        """
        Calculates the bulk energy dissipation for the given damage state.
        """
        return np.sum(self.params.Yc * self.params.dx * self.model.HD_bulk.get_Value(D_center))
    
    def get_bulk_dissipation_derivative(self, D_center):
        """
        Calculates the derivative of bulk energy dissipation with respect to damage.
        """
        return self.params.Yc * self.params.dx * self.model.HD_bulk.get_First_derivative(D_center)
    
    
    def assemble_lip_functional(self, d, u, returnall = False):
        """
        Assemble the functional to minimize
        """
        strain = self.get_strain(u)
       
        strain_energy = self.get_strain_energy(strain,d)
        
        bulk_dissipation = self.get_bulk_dissipation(d)
       

        lip_functional = strain_energy  + bulk_dissipation 
        if returnall:
            return lip_functional, strain_energy,  bulk_dissipation
        else:
            return lip_functional
        
    def assemble_jac_lip_functional(self, d, u, returnall = False):
        """
        Assemble the jacobian of the functional to minimize
        """
        strain = self.get_strain(u)
        d_strain_energy = self.get_strain_energy_derivative(strain, d)
        d_bulk_dissipation = self.get_bulk_dissipation_derivative(d)

        jac_clip_functional = (d_strain_energy + d_bulk_dissipation)
        if returnall:
            return jac_clip_functional, d_strain_energy, d_bulk_dissipation
        else:
            return jac_clip_functional

class Solver_Lip:
    """
    Coordinates the solution of the equilibrium equations and damage evolution.
    """
    def __init__(self, model, simulation_parameters):
        
        self.equilibrium_solver = EquilibriumSolver(model, simulation_parameters)
        self.functional = LipFunctional(model, simulation_parameters)
        self.params = simulation_parameters

    def equib_solver(self,d,bc):

        u_, F = self.equilibrium_solver.solve_equilibrium_ul(d, bc)
        return u_, F


    def  lip_functional(self, d, u_, low_bound):

        functional = lambda d:self.functional.assemble_lip_functional(d,u_)
        jacob = lambda d: self.functional.assemble_jac_lip_functional(d,u_)

        A = scipy.sparse.eye(self.params.N_v-2,self.params.N_v-1) - scipy.sparse.eye(self.params.N_v-2,self.params.N_v-1,1)           
        slopeconstrain = scipy.optimize.LinearConstraint(A, -self.params.dx/self.params.lc *np.ones(self.params.N_v-2), self.params.dx/self.params.lc*np.ones(self.params.N_v-2) )

        bounds = scipy.optimize.Bounds(low_bound,np.ones(len(d)))

        damage_predictor_opt = scipy.optimize.minimize(
            fun=functional,
            x0=d,
            bounds=bounds ,            
            method = 'SLSQP',         
            constraints=slopeconstrain,
            jac =jacob
            )
                
        if not damage_predictor_opt.success :
            print("Optimisation failed")
        return damage_predictor_opt


