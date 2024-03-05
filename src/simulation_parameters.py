import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import cos, sin
from matplotlib.ticker import FuncFormatter

class SimulationParameters:
    def __init__(self, E, Gc, sigc, L, Dm, N_increments, max_iter, a, gamma):
        self.E = E  # Young's Modulus (Pa)
        self.Gc = Gc  # Fracture Energy (N/m)
        self.sigc = sigc  # Stress Limit (N/m^2)
        self.L = L  # Length of the bar (m)
        self.Dm = Dm  # Bulk Damage parameter
        self.N_increments = N_increments  # Number of increments
        self.max_iter = max_iter  # Maximum number of iterations
        self.a = a  # G function - Co-efficients
        self.gamma = gamma

        # Derived parameters
        self.calculate_derived_parameters()

    def calculate_derived_parameters(self):
        self.wc = (2 * self.Gc) / self.sigc  # Critical Separation
        self.lc = self.L / 4  # Characteristic length
        self.lch = (self.E * self.Gc) / (self.sigc ** 2)  # Cohesive zone length
        self.lmb = self.lc / self.lch  # lambda
        self.c = (2 * self.gamma - 1) / 2  # h function - Co-efficients
        #self.c = 0
        self.b = self.c
        self.k = self.E / self.lch  # k
        self.Yc = (0.5 * self.sigc ** 2) / self.E  # Critical Energy release rate
        self.yc = (0.5 * self.sigc ** 2) / self.k  # Critical Energy release rate - cohesive
        self.N_lc = 10  # Nodes per characteristic length
        self.N_v = int(self.L / (self.lc / self.N_lc)) - 1  # Number of vertices/nodes
        self.N_elements = self.N_v - 1  # Number of elements
        self.x = np.linspace(0., self.L, self.N_v)  # grid parameter
        self.dx = self.L / self.N_elements  # Element size
        self.epsilon_0 = self.sigc / self.E  # Strain at the peak stress

    def get_len_mat(self, x): 
        return np.abs(np.broadcast_to(x, (x.shape[0], x.shape[0])) - np.repeat(x, x.shape[0]).reshape((-1, x.shape[0])))
