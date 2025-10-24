import numpy as np
from numpy import cos,sin
import akantu as aka

##################### g(d) ###########################################

class gd_cohesive_std():
    def __init__(self):
        """
        Initializes the gd_cohesive_std class.
        """
        return

    def get_value(self,d):
        """
        Computes the g(d) function value based on the cohesive damage d.
        """
        return (1./d) - 1

    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the g(d) function with respect to the cohesive damage d.
        """
        return -1/d**2

    def get_lmb_value(self,d):
        """
        Computes the (1)/(1 + g(d)) value, used to solve the equilbrium equation
        """
        return d

    def get_derivative_lmb_value(self,d):
        """
        Computes the derivative of the (1)/(1 + g(d)) value, used to solve the equilbrium equation
        """
        return  1

class gd_cohesive_twice_std():
    def __init__(self):
        """
        Initializes the gd_cohesive_twice_std class.
        """
        return

    def get_value(self,d):
        """
        Computes the g(d) function value based on the cohesive damage d.
        """
        return (2*(1-d))/(d)

    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the g(d) function with respect to the cohesive damage d.
        """
        return -2/d**2

    def get_lmb_value(self,d):
        """
        Computes the (1)/(1 + g(d)) value, used to solve the equilbrium equation
        """
        return d/(2 - d)

    def get_derivative_lmb_value(self,d):
        """
        Computes the derivative of the (1)/(1 + g(d)) value, used to solve the equilbrium equation
        """
        return  2/(2 - d)**2

class gd_cohesive_hyd():
    def __init__(self,parameters):
        """
        Initializes the gd_cohesive_hyd class.
        """
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the g(d) function value based on the cohesive damage d.
        """
        return (1-d)/((1-self.Dm)*d) 

    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the g(d) function with respect to the cohesive damage d.
        """
        return 1/(d**2*(self.Dm - 1))
    
    def get_second_derivative(self,d):
        return -2/(d**3*(self.Dm - 1))

    def get_lmb_value(self,d):
        """
        Computes the (1)/(1 + g(d)) value, used to solve the equilbrium equation
        """
        return d*(self.Dm - 1)/(self.Dm*d - 1)


    def get_derivative_lmb_value(self,d):
        """
        Computes the derivative of the (1)/(1 + g(d)) value, used to solve the equilbrium equation
        """
        return  (1 - self.Dm)/(self.Dm**2*d**2 - 2*self.Dm*d + 1)

################### h(d) for the CLIP model #############################################

class hd_cohesive_quad_4_terms():
    def __init__(self,parameters):
        self.beta = parameters.beta

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        return self.beta * d**2 + d

    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        return 2 * self.beta * d  + 1

class hd_cohesive_cubic_4_terms():
    def __init__(self,parameters):
        self.beta = parameters.beta
        self.beta_1 = parameters.beta_1

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        return self.beta * d**3 + self.beta_1 * d**2 + d

    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        return 3 * self.beta * d**2 + 2 * self.beta_1 * d + 1

class hd_cohesive_cos_D_squared_3_terms():
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        num = ((2 - 2*d)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1) + (3*d - 2)*(self.Dm*d - 1))
        den = (d*(self.Dm*d - 1) + (2 - 2*d)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1))
        return num/den
    
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        Dm = self.Dm
        alpha = self.alpha
        num = ((d*(Dm*d - 1) + (2 - 2*d)*(Dm*d + d*(cos(Dm**2*alpha*d**2) - 1) - 1))*(Dm*d + Dm*(3*d - 2) - 2*d*(cos(Dm**2*alpha*d**2) - 1) + (2*d - 2)*(2*Dm**2*alpha*d**2*sin(Dm**2*alpha*d**2) - Dm - cos(Dm**2*alpha*d**2) + 1) - 1) + ((2*d - 2)*(Dm*d + d*(cos(Dm**2*alpha*d**2) - 1) - 1) - (3*d - 2)*(Dm*d - 1))*(-2*d*(cos(Dm**2*alpha*d**2) - 1) + (2*d - 2)*(2*Dm**2*alpha*d**2*sin(Dm**2*alpha*d**2) - Dm - cos(Dm**2*alpha*d**2) + 1) + 1))
        den = (d*(Dm*d - 1) + (2 - 2*d)*(Dm*d + d*(cos(Dm**2*alpha*d**2) - 1) - 1))**2
        return num/den

class hd_cohesive_k_E_lch():
    def __init__(self):
            return

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        return 2*d/(2-d)
        
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        return -2/(d-2)**2

class hd_cohesive_hybrid():
    def __init__(self,parameters):
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        Dm = self.Dm
        num = d*(Dm**2*d**2*(Dm - 1) + 2*Dm*d*(1 - Dm) + Dm - 1)
        den = (Dm**4*d**5 - Dm**3*d**4*(2*Dm + 1) + Dm**2*d**3*(4*Dm + 1) - 5*Dm**2*d**2 + 3*Dm*d - 1)
        return num/den
    
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        Dm = self.Dm
        num = (Dm**2*d*(Dm*d - 2)*(Dm*d*(d - 1)*(Dm*d - 1) - d*(Dm - 1)*(Dm*d - 1) + d - 1)**2 + (Dm**2*d**2 - Dm*d + 1)**2*(-Dm*(d - 1)**2*(Dm*d - 1)**2 + Dm*(d - 1)**2 - d*(Dm - 1)*(Dm*d - 1)**2 + (Dm - 1)*(d - 1)*(Dm*d - 1)**2))
        den = ((Dm**2*d**2 - Dm*d + 1)**2*(Dm*d*(d - 1)*(Dm*d - 1) - d*(Dm - 1)*(Dm*d - 1) + d - 1)**2)
        return num/den

class hd_cohesive_hybrid_k():
    def __init__(self,parameters):
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        Dm = self.Dm
        num = 2*d*(Dm**2*d**2*(Dm - 1) + 2*Dm*d*(1 - Dm) + Dm - 1)
        den = (2*Dm**4*d**5 - 3*Dm**3*d**4*(Dm + 1) + 2*Dm**2*d**3*(3*Dm + 2) - 2*Dm*d**2*(4*Dm + 1) + d*(5*Dm + 1) - 2)
        return num/den
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        Dm = self.Dm
        num = 2*(-d*(Dm**2*d**2*(Dm - 1) + 2*Dm*d*(1 - Dm) + Dm - 1)*(10*Dm**4*d**4 - 12*Dm**3*d**3*(Dm + 1) + 6*Dm**2*d**2*(3*Dm + 2) - 4*Dm*d*(4*Dm + 1) + 5*Dm + 1) + (3*Dm**2*d**2*(Dm - 1) - 4*Dm*d*(Dm - 1) + Dm - 1)*(2*Dm**4*d**5 - 3*Dm**3*d**4*(Dm + 1) + 2*Dm**2*d**3*(3*Dm + 2) - 2*Dm*d**2*(4*Dm + 1) + d*(5*Dm + 1) - 2))
        den = (2*Dm**4*d**5 - 3*Dm**3*d**4*(Dm + 1) + 2*Dm**2*d**3*(3*Dm + 2) - 2*Dm*d**2*(4*Dm + 1) + d*(5*Dm + 1) - 2)**2
        return num/den

class hd_cohesive_hybrid_1():    
    def __init__(self,parameters):
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        Dm = self.Dm
        num = 2*d*(Dm**6*d**4 - 5*Dm**5*d**4 + 2*Dm**5*d**3 + 4*Dm**4*d**4 - 2*Dm**4*d**3 - Dm**4*d**2 + 4*Dm**3*d**2 - 3*Dm**2*d**2 - Dm + 1)
        den = (Dm**6*d**5 - 2*Dm**5*d**5 + Dm**4*d**5 - Dm**4*d**3 + Dm**3*d**3 + 2*Dm**3*d**2 - 2*Dm**2*d**2 - Dm*d - d + 2)
        return num/den
    
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        Dm = self.Dm
        num = 2*((8*Dm**4*d**3*(Dm**2 - 2*Dm + 1) + 9*Dm**2*d**2*(1 - Dm) - 4*Dm**2*d*(Dm**2 - 5*Dm + 4) - Dm - 5)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**3*d**3*(Dm**2 - 2*Dm + 1) + 2*Dm**2*d**2*(Dm**2 - 2*Dm + 1) + 2*Dm - d*(Dm**2 - 1) - 2)**2 + (5*Dm**4*d**4*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + 3*Dm**3*d**2*(-Dm**2 + 2*Dm - 1) + 4*Dm**2*d*(Dm**2 - 2*Dm + 1) - Dm**2 + 1)*(2*Dm**4*d**4*(-Dm**2 + 2*Dm - 1) + 3*Dm**2*d**3*(Dm - 1) + 2*Dm**2*d**2*(Dm**2 - 5*Dm + 4) + 2*Dm + d*(Dm + 5) - 8)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**3*d**3*(-Dm**2 + 2*Dm - 1) + 2*Dm**2*d**2*(Dm**2 - 2*Dm + 1) + 2*Dm - d*(Dm**2 - 1) - 2))
        den = (Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**3*d**3*(Dm**2 - 2*Dm + 1) + 2*Dm**2*d**2*(Dm**2 - 2*Dm + 1) + 2*Dm - d*(Dm**2 - 1) - 2)**3
        return num/den

    def get_second_derivative(self,d):
        Dm = self.Dm
        num = (6*((Dm**4*d**3*(-8*Dm**2 + 16*Dm - 8) + Dm**2*d**2*(9*Dm - 9) + 4*Dm**2*d*(Dm**2 - 5*Dm + 4) + Dm + 5)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**3*d**3*(Dm**2 - 2*Dm + 1) + 2*Dm**2*d**2*(Dm**2 - 2*Dm + 1) + 2*Dm - d*(Dm**2 - 1) - 2)**2 - (Dm**4*d**4*(5*Dm**3 - 15*Dm**2 + 15*Dm - 5) + Dm**3*d**2*(-3*Dm**2 + 6*Dm - 3) + 4*Dm**2*d*(Dm**2 - 2*Dm + 1) - Dm**2 + 1)*(Dm**4*d**4*(-2*Dm**2 + 4*Dm - 2) + Dm**2*d**3*(3*Dm - 3) + Dm**2*d**2*(2*Dm**2 - 10*Dm + 8) + 2*Dm + d*(Dm + 5) - 8)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**3*d**3*(-Dm**2 + 2*Dm - 1) + Dm**2*d**2*(2*Dm**2 - 4*Dm + 2) + 2*Dm - d*(Dm**2 - 1) - 2))*(Dm**4*d**4*(5*Dm**3 - 15*Dm**2 + 15*Dm - 5) + Dm**3*d**2*(-3*Dm**2 + 6*Dm - 3) + 4*Dm**2*d*(Dm**2 - 2*Dm + 1) - Dm**2 + 1) + 2*(Dm**2*(Dm**2*d**2*(24*Dm**2 - 48*Dm + 24) - 4*Dm**2 + 20*Dm - 18*d*(Dm - 1) - 16)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**3*d**3*(Dm**2 - 2*Dm + 1) + 2*Dm**2*d**2*(Dm**2 - 2*Dm + 1) + 2*Dm - d*(Dm**2 - 1) - 2)**2 + 2*Dm**2*(Dm**2*d**3*(10*Dm**3 - 30*Dm**2 + 30*Dm - 10) + 2*Dm**2 - 3*Dm*d*(Dm**2 - 2*Dm + 1) - 4*Dm + 2)*(Dm**4*d**4*(-2*Dm**2 + 4*Dm - 2) + Dm**2*d**3*(3*Dm - 3) + Dm**2*d**2*(2*Dm**2 - 10*Dm + 8) + 2*Dm + d*(Dm + 5) - 8)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**3*d**3*(-Dm**2 + 2*Dm - 1) + Dm**2*d**2*(2*Dm**2 - 4*Dm + 2) + 2*Dm - d*(Dm**2 - 1) - 2) - (Dm**4*d**3*(-8*Dm**2 + 16*Dm - 8) + Dm**2*d**2*(9*Dm - 9) + 4*Dm**2*d*(Dm**2 - 5*Dm + 4) + Dm + 5)*(Dm**4*d**4*(5*Dm**3 - 15*Dm**2 + 15*Dm - 5) + Dm**3*d**2*(-3*Dm**2 + 6*Dm - 3) + 4*Dm**2*d*(Dm**2 - 2*Dm + 1) - Dm**2 + 1)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**3*d**3*(-Dm**2 + 2*Dm - 1) + Dm**2*d**2*(2*Dm**2 - 4*Dm + 2) + 2*Dm - d*(Dm**2 - 1) - 2) + (5*Dm**4*d**4*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - 3*Dm**3*d**2*(Dm**2 - 2*Dm + 1) + 4*Dm**2*d*(Dm**2 - 2*Dm + 1) - Dm**2 + 1)**2*(Dm**4*d**4*(-2*Dm**2 + 4*Dm - 2) + Dm**2*d**3*(3*Dm - 3) + Dm**2*d**2*(2*Dm**2 - 10*Dm + 8) + 2*Dm + d*(Dm + 5) - 8))*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**3*d**3*(-Dm**2 + 2*Dm - 1) + Dm**2*d**2*(2*Dm**2 - 4*Dm + 2) + 2*Dm - d*(Dm**2 - 1) - 2))
        den = (Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**3*d**3*(Dm**2 - 2*Dm + 1) + 2*Dm**2*d**2*(Dm**2 - 2*Dm + 1) + 2*Dm - d*(Dm**2 - 1) - 2)**4
        return num/den

class hd_cohesive_hybrid_2():    
    def __init__(self,parameters):
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        Dm = self.Dm
        num = 2*d*(-Dm**4*d**4 - Dm**3*d**4 + 2*Dm**3*d**3 + Dm**3*d**2 + 2*Dm**2*d**4 - 2*Dm**2*d**3 - Dm*d**2 - Dm + 1)
        den = (Dm**3*d**3 + Dm**2*d**3 - 2*Dm**2*d**2 - 2*Dm*d**3 + 2*Dm*d**2 - Dm*d - d + 2)

        return num/den
    
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        Dm = self.Dm
        num = 4*(-Dm**7*d**7 - 2*Dm**6*d**7 + 4*Dm**6*d**6 + 3*Dm**5*d**7 - 2*Dm**5*d**5 - Dm**5*d**4 + 4*Dm**4*d**7 - 12*Dm**4*d**6 + 12*Dm**4*d**5 - 7*Dm**4*d**4 - 4*Dm**3*d**7 + 8*Dm**3*d**6 - 6*Dm**3*d**5 - 4*Dm**3*d**4 + 7*Dm**3*d**3 + 2*Dm**3*d**2 - 4*Dm**2*d**5 + 12*Dm**2*d**4 - 10*Dm**2*d**3 + 2*Dm**2*d**2 + 3*Dm*d**3 - 4*Dm*d**2 - Dm + 1)
        den = (Dm**6*d**6 + 2*Dm**5*d**6 - 4*Dm**5*d**5 - 3*Dm**4*d**6 + 2*Dm**4*d**4 - 4*Dm**3*d**6 + 12*Dm**3*d**5 - 12*Dm**3*d**4 + 8*Dm**3*d**3 + 4*Dm**2*d**6 - 8*Dm**2*d**5 + 6*Dm**2*d**4 + 4*Dm**2*d**3 - 7*Dm**2*d**2 + 4*Dm*d**4 - 12*Dm*d**3 + 10*Dm*d**2 - 4*Dm*d + d**2 - 4*d + 4)
        return num/den

class hd_cohesive_new_k():    
    def __init__(self,parameters):
        self.Dm = parameters.Dm

    def get_value(self,d):
        """
        Computes the h(d) function value based on the cohesive damage d.
        """
        Dm = self.Dm
        num = d*(Dm**6*d**4 - 4*Dm**5*d**4 + Dm**5*d**3 + 3*Dm**4*d**4 - Dm**4*d**3 - Dm**4*d**2 + 4*Dm**3*d**2 - 3*Dm**2*d**2 - Dm + 1)
        den = (Dm**6*d**5 - 2*Dm**5*d**5 + Dm**4*d**5 - Dm**4*d**3 + 2*Dm**3*d**3 + Dm**3*d**2 - Dm**2*d**3 - Dm**2*d**2 - Dm*d + 1)
        return num/den
    
    def get_first_derivative(self,d):
        """
        Computes the value of the first derivative of the h(d) function with respect to the cohesive damage d.
        """
        Dm = self.Dm
        num = (Dm*(5*Dm**3*d**4*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + 3*Dm*d**2*(-Dm**3 + 3*Dm**2 - 3*Dm + 1) + 2*Dm*d*(Dm**2 - 2*Dm + 1) - Dm + 1)*(Dm**4*d**4*(-Dm**2 + 2*Dm - 1) + Dm**2*d**2*(Dm**2 - 4*Dm + 3) + Dm + d*(Dm + 1) - 3)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**2*d**3*(-Dm**3 + 3*Dm**2 - 3*Dm + 1) + Dm**2*d**2*(Dm**2 - 2*Dm + 1) - Dm*d*(Dm - 1) + Dm - 1) + (4*Dm**4*d**3*(Dm**2 - 2*Dm + 1) - 2*Dm**2*d*(Dm**2 - 4*Dm + 3) - Dm - 1)*(Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**2*d**3*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**2*d**2*(Dm**2 - 2*Dm + 1) - Dm*d*(Dm - 1) + Dm - 1)**2)
        den = (Dm**4*d**5*(Dm**3 - 3*Dm**2 + 3*Dm - 1) - Dm**2*d**3*(Dm**3 - 3*Dm**2 + 3*Dm - 1) + Dm**2*d**2*(Dm**2 - 2*Dm + 1) - Dm*d*(Dm - 1) + Dm - 1)**3
        return num/den

    def get_second_derivative(self,d):
        Dm = self.Dm
        num = 2*Dm*(Dm**16*d**12 - 5*Dm**15*d**12 + 10*Dm**14*d**12 - 3*Dm**14*d**10 - 10*Dm**13*d**12 + 27*Dm**13*d**10 - 7*Dm**13*d**9 + 5*Dm**12*d**12 - 78*Dm**12*d**10 + 28*Dm**12*d**9 + 3*Dm**12*d**8 - Dm**11*d**12 + 102*Dm**11*d**10 - 52*Dm**11*d**9 - 24*Dm**11*d**8 + 3*Dm**11*d**7 - 63*Dm**10*d**10 + 48*Dm**10*d**9 + 108*Dm**10*d**8 - 36*Dm**10*d**7 + 15*Dm**9*d**10 - 7*Dm**9*d**9 - 210*Dm**9*d**8 + 99*Dm**9*d**7 + 10*Dm**9*d**6 - 20*Dm**8*d**9 + 177*Dm**8*d**8 - 102*Dm**8*d**7 - 66*Dm**8*d**6 + 9*Dm**8*d**5 + 10*Dm**7*d**9 - 54*Dm**7*d**8 + 27*Dm**7*d**7 + 154*Dm**7*d**6 - 51*Dm**7*d**5 + 18*Dm**6*d**7 - 150*Dm**6*d**6 + 69*Dm**6*d**5 + 21*Dm**6*d**4 - 9*Dm**5*d**7 + 52*Dm**5*d**6 - 30*Dm**5*d**5 - 39*Dm**5*d**4 + 48*Dm**4*d**4 - 27*Dm**4*d**3 + 3*Dm**4*d**2 + 3*Dm**3*d**5 - 30*Dm**3*d**4 + 28*Dm**3*d**3 - Dm**2*d**3 - 9*Dm**2*d**2 + 6*Dm**2*d + 6*Dm*d**2 - 6*Dm*d - Dm + 1)
        den = (Dm**18*d**15 - 6*Dm**17*d**15 + 15*Dm**16*d**15 - 3*Dm**16*d**13 - 20*Dm**15*d**15 + 18*Dm**15*d**13 + 3*Dm**15*d**12 + 15*Dm**14*d**15 - 45*Dm**14*d**13 - 15*Dm**14*d**12 + 3*Dm**14*d**11 - 6*Dm**13*d**15 + 60*Dm**13*d**13 + 30*Dm**13*d**12 - 21*Dm**13*d**11 - 6*Dm**13*d**10 + Dm**12*d**15 - 45*Dm**12*d**13 - 30*Dm**12*d**12 + 57*Dm**12*d**11 + 33*Dm**12*d**10 + 2*Dm**12*d**9 + 18*Dm**11*d**13 + 15*Dm**11*d**12 - 78*Dm**11*d**11 - 72*Dm**11*d**10 + 3*Dm**11*d**8 - 3*Dm**10*d**13 - 3*Dm**10*d**12 + 57*Dm**10*d**11 + 78*Dm**10*d**10 - 21*Dm**10*d**9 - 27*Dm**10*d**8 - 3*Dm**10*d**7 - 21*Dm**9*d**11 - 42*Dm**9*d**10 + 44*Dm**9*d**9 + 72*Dm**9*d**8 + 15*Dm**9*d**7 + Dm**9*d**6 + 3*Dm**8*d**11 + 9*Dm**8*d**10 - 36*Dm**8*d**9 - 84*Dm**8*d**8 - 21*Dm**8*d**7 + 6*Dm**8*d**6 + 12*Dm**7*d**9 + 45*Dm**7*d**8 + 6*Dm**7*d**7 - 33*Dm**7*d**6 - 9*Dm**7*d**5 - Dm**6*d**9 - 9*Dm**6*d**8 + 6*Dm**6*d**7 + 47*Dm**6*d**6 + 24*Dm**6*d**5 + 3*Dm**6*d**4 - 3*Dm**5*d**7 - 24*Dm**5*d**6 - 21*Dm**5*d**5 + 3*Dm**5*d**4 + 3*Dm**4*d**6 + 6*Dm**4*d**5 - 12*Dm**4*d**4 - 9*Dm**4*d**3 + 6*Dm**3*d**4 + 11*Dm**3*d**3 + 3*Dm**3*d**2 - 3*Dm**2*d**3 - 3*Dm*d + 1)
        return num/den


################# G(D) for the CLIP model ###############################################

class GD_bulk_cos_sin():
    """
    G(D) funtion:
    1-cos(aD)/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        num =  (1 - D)**2
        den =  ((1 - D)**2 + (self.alpha * sin(self.alpha * D) * (1 - D) + 1 - cos(self.alpha * D)) / (self.gamma))
        return num/den
    
    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        gamma = self.gamma
        alpha = self.alpha
        num =  gamma*(D - 1)*(-2*alpha*(D - 1)*sin(D*alpha) + 2*gamma*(D - 1)**2 + (D - 1)*(alpha**2*(D - 1)*cos(D*alpha) + 2*gamma*(1 - D)) - 2*cos(D*alpha) + 2)
        den = (-alpha*(D - 1)*sin(D*alpha) + gamma*(D - 1)**2 - cos(D*alpha) + 1)**2
        return num/den

class GD_bulk_D_squared():
    """
    a(1-(1-D)**2)/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        num =  ((1-D)**2)
        den = ((1-D)**2 + ((self.alpha*(-D*(D-2)+2*(D-1)**2)-(2*self.alpha*(1-D)**2)))/(self.gamma))
        return num/den
    
    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        gamma = self.gamma
        alpha = self.alpha
        num = 2*gamma*(D - 1)*(-2*alpha*(D - 1)**2 - alpha*(D*(D - 2) - 2*(D - 1)**2) + gamma*(D - 1)**2 + (D - 1)*(alpha*(D - 1) + gamma*(1 - D)))
        den = (-2*alpha*(D - 1)**2 - alpha*(D*(D - 2) - 2*(D - 1)**2) + gamma*(D - 1)**2)**2
        return num/den

class GD_bulk_D():
    """
    aD/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        num = ((1-D)**2)
        den = ((1-D)**2 + (self.alpha-self.alpha*(1-D)**2)/(self.gamma))
        return num/den

    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        num =2*self.gamma*(D - 1)*(-self.alpha*(D - 1)**2 + self.alpha + self.gamma*(D - 1)**2 + (D - 1)*(self.alpha*(D - 1) + self.gamma*(1 - D)))
        den = (-self.alpha*(D - 1)**2 + self.alpha + self.gamma*(D - 1)**2)**2
        return num/den

class GD_bulk_cos_D_squared():
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        gamma = self.gamma
        alpha = self.alpha
        num =-gamma*(D - 1)**2
        den =(D*(cos(D**2*alpha) - 1) - gamma*(D - 1)**2 + (D - 1)*(2*D**2*alpha*sin(D**2*alpha) - cos(D**2*alpha) + 1))
        return num/den

    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        gamma = self.gamma
        alpha = self.alpha
        num = 2*gamma*(D - 1)*(-D*(cos(D**2*alpha) - 1) + gamma *(D - 1)**2 + (1 - D)*(2*D**2*alpha*sin(D**2*alpha) - cos(D**2*alpha) + 1) - (D - 1)**2*(-D*alpha*(2*D**2*alpha*cos(D**2*alpha) + 3*sin(D**2*alpha)) + gamma ))
        den = (D*(cos(D**2*alpha) - 1) - gamma *(D - 1)**2 + (D - 1)*(2*D**2*alpha*sin(D**2*alpha) - cos(D**2*alpha) + 1))**2
        return num/den
    
class GD_bulk_hybrid():
    def __init__(self,parameters):
        self.Dm = parameters.Dm
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        gamma = self.gamma
        Dm = self.Dm
        num =-gamma*(D - 1)**2
        den = (D*Dm*(D - 2) - gamma*(D - 1)**2)
        return num/den

    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        gamma = self.gamma
        Dm = self.Dm
        num = (2*gamma*Dm*(D-1))
        den = ((Dm-gamma)*D**2+(2*gamma-2*Dm)*D-gamma)**2
        return num/den

class GD_bulk_hybrid_1():
    def __init__(self,parameters):
        self.Dm = parameters.Dm
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        gamma = self.gamma
        Dm = self.Dm
        num = gamma*(D**2 - 1)**2
        den = (2*D*Dm + gamma*(D**2 - 1)**2)
        return num/den

    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        gamma = self.gamma
        Dm = self.Dm
        num = 2*gamma*(D**2 - 1)*(2*D*(2*D*Dm + gamma*(D**2 - 1)**2) - (D**2 - 1)*(2*D*gamma*(D**2 - 1) + Dm))
        den = (2*D*Dm + gamma*(D**2 - 1)**2)**2
        return num/den

class GD_bulk_hybrid_2():
    def __init__(self,parameters):
        self.Dm = parameters.Dm
        self.gamma = parameters.gamma

    def get_value(self,D):
        """
        Computes the G(D) function value based on the bulk damage D.
        """
        gamma = self.gamma
        Dm = self.Dm
        num = gamma*(D**2 - 1)**2
        den = (2*D + gamma*(D**2 - 1)**2)
        return num/den

    def get_first_derivative(self,D):
        """
        Computes the value of the first derivative of the G(D) function with respect to the bulk damage D.
        """
        gamma = self.gamma
        Dm = self.Dm
        num = 2*gamma*(D**2 - 1)*(2*D*(2*D + gamma*(D**2 - 1)**2) - (D**2 - 1)*(2*D*gamma*(D**2 - 1) + 1))
        den = (2*D + gamma*(D**2 - 1)**2)**2
        return num/den

################### H(D) for the CLIP model #############################################

class HD_bulk_cos_sin_4_terms():
    """
    1-cos(aD)/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm
        self.beta = parameters.beta 
        self.beta_1 = parameters.beta_1

    def get_value(self,D):
        Dm = self.Dm
        alpha = self.alpha
        num = 4*Dm**4*(Dm**2*(D - 1)**2 + 2*(D - Dm)**2*(-D*alpha*sin(D*alpha) + alpha*sin(D*alpha) - cos(D*alpha) + 1)) - (2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))**2*(3*D**2*self.beta + 2*D*Dm*self.beta_1 + Dm**2)
        den = (2*Dm**2*self.gamma*(2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))**2)
        return num/den
 
    def get_first_derivative(self,D):
        Dm = self.Dm
        alpha = self.alpha
        num = ((2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))*(4*Dm**4*(Dm**2*(D - 1) + alpha**2*(1 - D)*(D - Dm)**2*cos(D*alpha) + 2*(-D + Dm)*(D*alpha*sin(D*alpha) - alpha*sin(D*alpha) + cos(D*alpha) - 1)) - (3*D*self.beta + Dm*self.beta_1)*(2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))**2 + (2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))*(3*D**2*self.beta + 2*D*Dm*self.beta_1 + Dm**2)*(2*D*alpha*sin(D*alpha) - Dm*(2*D - 2*Dm + 2*alpha*sin(D*alpha) - 1) - 2*cos(D*alpha) + 2)) + (4*Dm**4*(Dm**2*(D - 1)**2 + 2*(D - Dm)**2*(-D*alpha*sin(D*alpha) + alpha*sin(D*alpha) - cos(D*alpha) + 1)) - (2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))**2*(3*D**2*self.beta + 2*D*Dm*self.beta_1 + Dm**2))*(2*D*alpha*sin(D*alpha) - Dm*(2*D - 2*Dm + 2*alpha*sin(D*alpha) - 1) - 2*cos(D*alpha) + 2))
        den = (Dm**2*self.gamma*(2*D*(cos(D*alpha) - 1) + Dm*((D - 1)*(D - 2*Dm) - 2*cos(D*alpha) + 2))**3)
        return num/den

class HD_bulk_D_squared_4_terms():
    """
    a(1-(1-D)**2)/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm
        self.beta = parameters.beta 
        self.beta_1 = parameters.beta_1

    def get_value(self,D):
        num = (8*D*self.Dm**3*self.alpha*(2 - D)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta*(-D**2 + 2*D - 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(D**2 - 2*D + 1) + self.Dm*(-D**2 + 2*D - 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        den = (2*self.Dm*self.gamma*(D**2 - 2*D + 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        return num/den
    
    def get_first_derivative(self,D):
        num = ((D - 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(-D**2 + 2*D - 1) + self.Dm*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2) + (D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(-8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm) - 8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)*(D - self.Dm)**2 - 4*D*self.Dm**3*self.alpha*(D - 1)**2*(D - self.Dm)**2 - 2*D*self.beta_1*(D - 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 - 2*D*self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1)) + 4*self.Dm**5*(D - 1)**3 + 4*self.Dm**5*(D - 1)*(D**2 - 2*D + 1) - 4*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm)**2 - self.Dm*(D - 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 - self.Dm*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1)) - self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2) + (D**2 - 2*D + 1)*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1))*(8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(-D**2 + 2*D - 1) + self.Dm*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2))
        den = (self.Dm*self.gamma*(D**2 - 2*D + 1)**2*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**3)
        return num/den

class HD_bulk_D_4_terms():
    """
    aD/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm
        self.beta = parameters.beta 
        self.beta_1 = parameters.beta_1

    def get_value(self,D):
        num = (4*self.Dm**3*(2*self.alpha*(D - self.Dm)**2 + (D - 1)**2*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - (2*D*self.beta + self.Dm)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        den = (2*self.Dm*self.gamma*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        return num/den        
            
    def get_first_derivative(self,D):
        Dm = self.Dm
        alpha = self.alpha
        num = (-(4*Dm**3*(2*alpha*(D - Dm)**2 + (D - 1)**2*(Dm**2 - 2*alpha*(D - Dm)**2)) - (2*D*self.beta_1 + Dm)*(2*D**3*alpha - D*Dm*(2*D*alpha + D - 1) + 2*Dm**2*(D - 1))**2)*(6*D**2*alpha - D*Dm*(2*alpha + 1) + 2*Dm**2 - Dm*(2*D*alpha + D - 1)) + (2*D**3*alpha - D*Dm*(2*D*alpha + D - 1) + 2*Dm**2*(D - 1))*(4*Dm**3*(-2*alpha*(D - 1)**2*(D - Dm) + 2*alpha*(D - Dm) + (D - 1)*(Dm**2 - 2*alpha*(D - Dm)**2)) - self.beta_1*(2*D**3*alpha - D*Dm*(2*D*alpha + D - 1) + 2*Dm**2*(D - 1))**2 - (2*D*self.beta_1 + Dm)*(2*D**3*alpha - D*Dm*(2*D*alpha + D - 1) + 2*Dm**2*(D - 1))*(6*D**2*alpha - D*Dm*(2*alpha + 1) + 2*Dm**2 - Dm*(2*D*alpha + D - 1))))
        den = (Dm*self.gamma*(2*D**3*alpha - D*Dm*(2*D*alpha + D - 1) + 2*Dm**2*(D - 1))**3)
        return num/den
    
class HD_bulk_hybrid():
    """
    aD/(1-D)
    """
    def __init__(self,parameters):        
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm
        

    def get_value(self,D):
        num = -D*self.Dm*(D - 2)
        den = (self.gamma*(D**2 - D + 1)**2)
        return num/den        
            
    def get_first_derivative(self,D):
        Dm = self.Dm
        gamma = self.gamma
        num = 2*Dm*(D*(D - 2)*(2*D - 1) - (D - 1)*(D**2 - D + 1))
        den =(gamma*(D**2 - D + 1)**3)
        return num/den

class HD_bulk_hybrid_1():
    def __init__(self,parameters):        
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm
        

    def get_value(self,D):

        num = 2*D*self.Dm
        den = (self.gamma*(D**2*self.Dm - D**2 + 1)**2)
        return num/den        
            
    def get_first_derivative(self,D):
        Dm = self.Dm
        gamma = self.gamma
        num = 2*Dm*(D**2*Dm - 4*D**2*(Dm - 1) - D**2 + 1)
        den = (gamma*(D**2*Dm - D**2 + 1)**3)
        return num/den

class HD_bulk_hybrid_2():
    def __init__(self,parameters):        
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm

    def get_value(self,D):

        num = 2*D*self.Dm
        den = (self.gamma)
        return num/den        
            
    def get_first_derivative(self,D):
        Dm = self.Dm
        gamma = self.gamma
        num = 2
        den = gamma
        return num/den


################################################################

class Functions_4_terms:
    def __init__(self,parameters):
        self.damage_function = parameters.damage_function
        self.gd_cohesive = gd_cohesive_std()
        if self.damage_function == 'cos_sin':
            self.hd_cohesive = hd_cohesive_cubic_4_terms(parameters)
            self.GD_bulk = GD_bulk_cos_sin(parameters)
            self.HD_bulk = HD_bulk_cos_sin_4_terms(parameters)

        elif self.damage_function == 'D_squared':
            self.hd_cohesive = hd_cohesive_quad_4_terms(parameters)
            self.GD_bulk = GD_bulk_D_squared(parameters)
            self.HD_bulk = HD_bulk_D_squared_4_terms(parameters)

        elif self.damage_function == 'D_std':
            self.hd_cohesive = hd_cohesive_quad_4_terms(parameters)
            self.GD_bulk = GD_bulk_D(parameters)
            self.HD_bulk = HD_bulk_D_4_terms(parameters)

        elif self.damage_function == 'hybrid':
            self.gd_cohesive = gd_cohesive_hyd(parameters)
            self.hd_cohesive = hd_cohesive_hybrid(parameters)
            self.GD_bulk = GD_bulk_hybrid(parameters)
            self.HD_bulk = HD_bulk_hybrid(parameters)

        elif self.damage_function == 'hybrid_k':
            self.gd_cohesive = gd_cohesive_hyd(parameters)
            self.hd_cohesive = hd_cohesive_hybrid_k(parameters)
            self.GD_bulk = GD_bulk_hybrid(parameters)
            self.HD_bulk = HD_bulk_hybrid(parameters)
        
        elif self.damage_function == 'hybrid_1':
            self.gd_cohesive = gd_cohesive_hyd(parameters)
            self.hd_cohesive = hd_cohesive_hybrid_1(parameters)
            self.GD_bulk = GD_bulk_hybrid_1(parameters)
            self.HD_bulk = HD_bulk_hybrid_1(parameters)

        elif self.damage_function == 'hybrid_2':
            self.gd_cohesive = gd_cohesive_hyd(parameters)
            self.hd_cohesive = hd_cohesive_hybrid_2(parameters)
            self.GD_bulk = GD_bulk_hybrid_2(parameters)
            self.HD_bulk = HD_bulk_hybrid_2(parameters)
        
        elif self.damage_function == 'new_k':
            self.gd_cohesive = gd_cohesive_hyd(parameters)
            self.hd_cohesive = hd_cohesive_new_k(parameters)
            self.GD_bulk = GD_bulk_hybrid_1(parameters)
            self.HD_bulk = HD_bulk_hybrid_1(parameters)

        else :
            raise ValueError('parameters.damage_function should be one of "cos_sin" or "D_squared" or "D_std" ')

################################################################

class Functions_3_terms:
    def __init__(self,parameters):
        self.damage_function = parameters.damage_function

        if self.damage_function == 'cos_sin_D_squared':
            self.gd_cohesive = gd_cohesive_twice_std()
            self.hd_cohesive = hd_cohesive_cos_D_squared_3_terms(parameters)
            self.GD_bulk = GD_bulk_cos_D_squared(parameters)
        else :
            raise ValueError('parameters.damage_function should be one of "cos_sin_D_squared" ')
        
#################### h(d) for the CZM model ############################################

class hd_cohesive_czm():
    """
    h(d) for the CZM model
    """
    def __init__(self):
        return

    def get_value(self,d):
        return d

    def get_first_derivative(self,d):
        return 1

class GD_bulk_czm():
    """
    G(D) for the CZM model
    """
    def __init__(self):
        return

    def get_value(self,D):
        return np.ones_like(D)

    def get_first_derivative(self,D):
        return np.zeros_like(D)

class Functions_CZM:

    def __init__(self,parameters):
        self.damage_function = parameters.damage_function
    
        if self.damage_function == 'CZM':
            self.gd_cohesive = gd_cohesive_std()
            self.hd_cohesive = hd_cohesive_czm()
            self.GD_bulk = GD_bulk_czm()
      
#################### G(D) for the LIP model ################################################

class GD_bulk_lip():
    """
    G(D) for the LIP model
    """
    def __init__(self,parameters):
        #self.alpha = parameters.alpha
        self.alpha = np.pi/4
        self.gamma = parameters.gamma

    def get_value(self,D):
        # num = (1.-D)**2
        # den =((1.-D)**2 + (self.alpha*np.sin(self.alpha*D)*(1.-D)+1.-np.cos(self.alpha*D))*(1/self.gamma))
        num = self.gamma * (1- D**2)**2
        den = self.gamma*(1 - D**2)**2 + 2*D
        return num/den
    
    def get_first_derivative(self,D):
        gamma = self.gamma
        alpha = self.alpha
        num = gamma*((D - 1.0)**2*(alpha **2*(D - 1.0)*cos(D*alpha ) + gamma*(2.0 - 2*D)) + (2*D - 2.0)*(-alpha *(D - 1.0)*sin(D*alpha ) + gamma*(D - 1.0)**2 - cos(D*alpha ) + 1.0)) 
        den = (-alpha *(D - 1.0)*sin(D*alpha ) + gamma*(D - 1.0)**2 - cos(D*alpha ) + 1.0)**2

        num = 2*gamma*(D**2 - 1)*(2*D*(2*D + gamma*(D**2 - 1)**2) - (D**2 - 1)*(2*D*gamma*(D**2 - 1) + 1))
        den = (2*D + gamma*(D**2 - 1)**2)**2
        return num/den

################# H(D) for the LIP model ###############################################

class HD_bulk_lip():
    """
    H(D) for the LIP model
    """
    def __init__(self,parameters):
        #self.alpha = parameters.alpha
        self.alpha = np.pi/4
        self.gamma = parameters.gamma
        self.lc = parameters.lc
        self.wc = parameters.wc
        self.sigc = parameters.sigc
        self.E = parameters.E

    def get_value(self,D):
        # num = (1/self.gamma)*(self.alpha*np.sin(self.alpha*D)*(1.-D)+(1.-np.cos(self.alpha*D))) 
        # den = ((1.-D)+1-np.cos(self.alpha*D))**2
        num = (D * self.E * self.wc)
        den = self.sigc*self.lc
        return num/den
    
    def get_first_derivative(self,D):
        gamma = self.gamma
        alpha = self.alpha
        num = -(0.25*alpha**2*(D - 1.0)*(0.5*D + 0.5*cos(D*alpha) - 1)*cos(D*alpha) + 0.25*(alpha*sin(D*alpha) - 1)*(alpha*(D - 1.0)*sin(D*alpha) + cos(D*alpha) - 1.0))
        den = (gamma*(0.5*D + 0.5*cos(D*alpha) - 1)**3)

        num = 2
        den = gamma

        return num/den

################################################################

class Functions_Lip:

    def __init__(self,parameters):
        self.damage_function = parameters.damage_function

        if self.damage_function == 'LIP':
            self.GD_bulk = GD_bulk_lip(parameters)
            self.HD_bulk = HD_bulk_lip(parameters)

################################################################

class Functions_explicit_czm :

    def __init__(self):
        self.gd_cohesive = gd_cohesive_std()
        self.hd_cohesive = hd_cohesive_k_E_lch()

class Functions_explicit_clip :
    def __init__(self, parameters):
        self.gd_cohesive = gd_cohesive_hyd(parameters)
        self.hd_cohesive = hd_cohesive_hybrid_k(parameters)
        self.GD_bulk = GD_bulk_hybrid(parameters)
        self.HD_bulk = HD_bulk_hybrid(parameters)


