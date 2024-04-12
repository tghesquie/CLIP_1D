import numpy as np

class Simulation_Parameters:
    
    def __init__(self, E, Gc, sigc, L, Dm, alpha, beta, he, choice ,N_increments, max_iter ):

        self.E = E  # Young's Modulus (Pa)
        self.Gc = Gc  # Fracture Energy (N/m)
        self.sigc = sigc  # Stress Limit (N/m^2)
        self.L = L  # Length of the bar (m)

        self.Dm = Dm  # Bulk Damage parameter
        self.N_increments = N_increments  # Number of increments
        self.max_iter = max_iter  # Maximum number of iterations
        self.alpha = alpha  # G function - Co-efficients
        self.beta = beta # H- function - Co-efficients
        self.alpha_values = None
        self.Dm_values = None
        self.choice_values = None

        self.choice = choice # Choice for the G and H function
        self.he = he # Nodes per characteristic length
        
        # Derived parameters
        self.calculate_derived_parameters()

    def calculate_derived_parameters(self):

        self.wc = (2 * self.Gc) / self.sigc  # Critical Separation
        self.lc = self.L / 4  # Characteristic length
        self.lch = (self.E * self.Gc) / (self.sigc ** 2)  # Cohesive zone length
        self.gamma = self.lc / self.lch  # lambda

        self.beta_1 = self.beta
        self.k = self.E / self.lch  # k

        self.Yc = (0.5 * self.sigc ** 2) / self.E  # Critical Energy release rate
        self.yc = (0.5 * self.sigc ** 2) / self.k  # Critical Energy release rate - cohesive

        self.N_v = int(self.L / (self.lc / self.he)) - 1  # Number of vertices/nodes
        self.N_elements = self.N_v - 1  # Number of elements
        self.x = np.linspace(0., self.L, self.N_v)  # grid parameter
        self.dx = self.L / self.N_elements  # Element size

        self.epsilon_0 = self.sigc / self.E  # Strain at the peak stress

    def get_len_mat(self, x): 
        return np.abs(np.broadcast_to(x, (x.shape[0], x.shape[0])) - np.repeat(x, x.shape[0]).reshape((-1, x.shape[0])))
    
    def pure_czm(self):
        exact_f = [0,self.sigc, 0 ]
        exact_inc = [0,(self.sigc * self.L)/self.E,self.wc]

        return exact_inc, exact_f