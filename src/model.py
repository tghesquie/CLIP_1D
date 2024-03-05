import numpy as np
from scipy.optimize import approx_fprime
from numpy import cos, sin
from simulation_parameters import SimulationParameters

class Model:

    def __init__(self, parameters):
        self.a = parameters.a
        self.c = parameters.c
        self.b = parameters.b
        self.lmb = parameters.lmb
        self.Dm = parameters.Dm

    def gd_cohesive(self, d):
        """Cohesive softening function."""
        return (1. / d) - 1.

    def dgd_cohesive_dd(self, d):
        """Derivative of the cohesive softening function."""
        return -1. / d**2

    def hd_cohesive(self, d):
        """Cohesive energy dissipation function."""
        return self.c * d**3 + self.b * d**2 + d

    def dhd_cohesive_dd(self, d):
        """Derivative of the cohesive energy dissipation function."""
        return 3 * self.c * d**2 + 2 * self.b * d + 1

    def GD_bulk(self, D):
        """Bulk softening function."""
        return (1 - D)**2 / ((1 - D)**2 + (self.a * sin(self.a * D) * (1 - D) + 1 - cos(self.a * D)) / (self.lmb))

    def dGD_bulk_dD(self, D):
        """Derivative of the bulk softening function."""
        tmp = ((self.a * (1 - D) * sin(self.a * D) - cos(self.a * D) + 1) / self.lmb + (1 - D)**2)
        return -((1 - D)**2 * ((self.a**2 * (1 - D) * cos(self.a * D)) / self.lmb - 2 * (1 - D))) / tmp**2 - (2 * (1 - D)) / tmp

    def HD_bulk(self, D):
        """Bulk energy dissipation function."""
        num = (4 * self.Dm**4 * (self.Dm**2 * (D - 1)**2 + 2 * (D - self.Dm)**2 * (-D * self.a * sin(D * self.a) + self.a * sin(D * self.a) - cos(D * self.a) + 1)) - (2 * D * (cos(D * self.a) - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos(D * self.a) + 2))**2 * (3 * D**2 * self.c + 2 * D * self.Dm * self.b + self.Dm**2))
        den = (2 * self.Dm**2 * self.lmb * (2 * D * (cos(D * self.a) - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos(D * self.a) + 2))**2)
        return num / den

    def dHD_bulk_dD(self, D):
        """Derivative of the bulk energy dissipation function."""
        cos_aD = cos(self.a * D)
        sin_aD = sin(self.a * D)
        num = ((2 * D * (cos_aD - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos_aD + 2)) * (4 * self.Dm**4 * (self.Dm**2 * (D - 1) + self.a**2 * (1 - D) * (D - self.Dm)**2 * cos_aD + 2 * (-D + self.Dm) * (D * self.a * sin_aD - self.a * sin_aD + cos_aD - 1)) - (3 * D * self.c + self.Dm * self.b) * (2 * D * (cos_aD - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos_aD + 2))**2 + (2 * D * (cos_aD - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos_aD + 2)) * (3 * D**2 * self.c + 2 * D * self.Dm * self.b + self.Dm**2) * (2 * D * self.a * sin_aD - self.Dm * (2 * D - 2 * self.Dm + 2 * self.a * sin_aD - 1) - 2 * cos_aD + 2)) + (4 * self.Dm**4 * (self.Dm**2 * (D - 1)**2 + 2 * (D - self.Dm)**2 * (-D * self.a * sin_aD + self.a * sin_aD - cos_aD + 1)) - (2 * D * (cos_aD - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos_aD + 2))**2 * (3 * D**2 * self.c + 2 * D * self.Dm * self.b + self.Dm**2)) * (2 * D * self.a * sin_aD - self.Dm * (2 * D - 2 * self.Dm + 2 * self.a * sin_aD - 1) - 2 * cos_aD + 2))
        den = (self.Dm**2 * self.lmb * (2 * D * (cos_aD - 1) + self.Dm * ((D - 1) * (D - 2 * self.Dm) - 2 * cos_aD + 2))**3)
        return num / den

