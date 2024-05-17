
import numpy as np
import scipy

from numpy import cos,sin

################################################################

class gd_cohesive_std():
    def __init__(self):
        return
    
    def get_Value(self,d):
        return (1./d) - 1
    
    def get_First_derivative(self,d):
        return -1/d**2
    
    def get_lmb_value(self,d):
        return d
    
    def get_derivative_lmb_value(self,d):
        return  1

class gd_cohesive_twice_std():
    def __init__(self):
        return
    
    def get_Value(self,d):
        return (2*(1-d))/(d)
    
    def get_First_derivative(self,d):
        return -2/d**2
    
    def get_lmb_value(self,d):
        return d/(2 - d)
    
    def get_derivative_lmb_value(self,d):
        return  2/(2 - d)**2

################################################################
"""
h(d) for the CLIP model
"""
class hd_cohesive_quad_4_terms():
    def __init__(self,parameters):
        self.beta = parameters.beta 
        
    def get_Value(self,d):
        return self.beta * d**2 + d
    
    def get_First_derivative(self,d):
        return 2 * self.beta * d  + 1
    
class hd_cohesive_cubic_4_terms():
    def __init__(self,parameters):
        self.beta = parameters.beta 
        self.beta_1 = parameters.beta_1

    def get_Value(self,d):
        return self.beta * d**3 + self.beta_1 * d**2 + d
    
    def get_First_derivative(self,d):
        return 3 * self.beta * d**2 + 2 * self.beta_1 * d + 1

class hd_cohesive_cos_D_squared_3_terms():
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.Dm = parameters.Dm

    def get_Value(self,d):
        num = ((2 - 2*d)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1) + (3*d - 2)*(self.Dm*d - 1))
        den = (d*(self.Dm*d - 1) + (2 - 2*d)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1))
        return num/den
    
    def get_First_derivative(self,d):
        num = ((d*(self.Dm*d - 1) + (2 - 2*d)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1))*(self.Dm*d + self.Dm*(3*d - 2) - 2*d*(cos(self.Dm**2*self.alpha*d**2) - 1) + (2*d - 2)*(2*self.Dm**2*self.alpha*d**2*sin(self.Dm**2*self.alpha*d**2) - self.Dm - cos(self.Dm**2*self.alpha*d**2) + 1) - 1) + ((2*d - 2)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1) - (3*d - 2)*(self.Dm*d - 1))*(-2*d*(cos(self.Dm**2*self.alpha*d**2) - 1) + (2*d - 2)*(2*self.Dm**2*self.alpha*d**2*sin(self.Dm**2*self.alpha*d**2) - self.Dm - cos(self.Dm**2*self.alpha*d**2) + 1) + 1))
        den = (d*(self.Dm*d - 1) + (2 - 2*d)*(self.Dm*d + d*(cos(self.Dm**2*self.alpha*d**2) - 1) - 1))**2
        return num/den
    
################################################################
"""
G(D) for the CLIP model
"""

class GD_bulk_cos_sin():
    """
    G(D) funtion 
    1-cos(aD)/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_Value(self,D):
        num =  (1 - D)**2 
        den =  ((1 - D)**2 + (self.alpha * sin(self.alpha * D) * (1 - D) + 1 - cos(self.alpha * D)) / (self.gamma))
        return num/den
    
    def get_First_derivative(self,D):
        num =  self.gamma*(D - 1)*(-2*self.alpha*(D - 1)*sin(D*self.alpha) + 2*self.gamma*(D - 1)**2 + (D - 1)*(self.alpha**2*(D - 1)*cos(D*self.alpha) + 2*self.gamma*(1 - D)) - 2*cos(D*self.alpha) + 2)
        den = (-self.alpha*(D - 1)*sin(D*self.alpha) + self.gamma*(D - 1)**2 - cos(D*self.alpha) + 1)**2
        return num/den

class GD_bulk_D_squared():
    """
    a(1-(1-D)**2)/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_Value(self,D):
        num =  ((1-D)**2)
        den = ((1-D)**2 + ((self.alpha*(-D*(D-2)+2*(D-1)**2)-(2*self.alpha*(1-D)**2)))/(self.gamma))
        return num/den
    
    def get_First_derivative(self,D):
        num = 2*self.gamma*(D - 1)*(-2*self.alpha*(D - 1)**2 - self.alpha*(D*(D - 2) - 2*(D - 1)**2) + self.gamma*(D - 1)**2 + (D - 1)*(self.alpha*(D - 1) + self.gamma*(1 - D)))
        den = (-2*self.alpha*(D - 1)**2 - self.alpha*(D*(D - 2) - 2*(D - 1)**2) + self.gamma*(D - 1)**2)**2
        return num/den

class GD_bulk_D():
    """
    aD/(1-D)
    """
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_Value(self,D):
        num = ((1-D)**2)
        den = ((1-D)**2 + (self.alpha-self.alpha*(1-D)**2)/(self.gamma))  
        return num/den
    
    def get_First_derivative(self,D):
        num =2*self.gamma*(D - 1)*(-self.alpha*(D - 1)**2 + self.alpha + self.gamma*(D - 1)**2 + (D - 1)*(self.alpha*(D - 1) + self.gamma*(1 - D)))
        den = (-self.alpha*(D - 1)**2 + self.alpha + self.gamma*(D - 1)**2)**2
        return num/den
    
class GD_bulk_cos_D_squared():
        def __init__(self,parameters):
            self.alpha = parameters.alpha
            self.gamma = parameters.gamma

        def get_Value(self,D):               
            num =-self.gamma*(D - 1)**2
            den =(D*(cos(D**2*self.alpha) - 1) - self.gamma*(D - 1)**2 + (D - 1)*(2*D**2*self.alpha*sin(D**2*self.alpha) - cos(D**2*self.alpha) + 1))
            return num/den

        def get_First_derivative(self,D):
            num = 2*self.gamma*(D - 1)*(-D*(cos(D**2*self.alpha) - 1) + self.gamma *(D - 1)**2 + (1 - D)*(2*D**2*self.alpha*sin(D**2*self.alpha) - cos(D**2*self.alpha) + 1) - (D - 1)**2*(-D*self.alpha*(2*D**2*self.alpha*cos(D**2*self.alpha) + 3*sin(D**2*self.alpha)) + self.gamma ))
            den = (D*(cos(D**2*self.alpha) - 1) - self.gamma *(D - 1)**2 + (D - 1)*(2*D**2*self.alpha*sin(D**2*self.alpha) - cos(D**2*self.alpha) + 1))**2
            return num/den

################################################################
"""
H(D) for the CLIP model
"""

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

    def get_Value(self,D):
        num = 4*self.Dm**4*(self.Dm**2*(D - 1)**2 + 2*(D - self.Dm)**2*(-D*self.alpha*sin(D*self.alpha) + self.alpha*sin(D*self.alpha) - cos(D*self.alpha) + 1)) - (2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2*(3*D**2*self.beta + 2*D*self.Dm*self.beta_1 + self.Dm**2)
        den = (2*self.Dm**2*self.gamma*(2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2)
        return num/den
    
    def get_First_derivative(self,D):
        num = ((2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))*(4*self.Dm**4*(self.Dm**2*(D - 1) + self.alpha**2*(1 - D)*(D - self.Dm)**2*cos(D*self.alpha) + 2*(-D + self.Dm)*(D*self.alpha*sin(D*self.alpha) - self.alpha*sin(D*self.alpha) + cos(D*self.alpha) - 1)) - (3*D*self.beta + self.Dm*self.beta_1)*(2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2 + (2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))*(3*D**2*self.beta + 2*D*self.Dm*self.beta_1 + self.Dm**2)*(2*D*self.alpha*sin(D*self.alpha) - self.Dm*(2*D - 2*self.Dm + 2*self.alpha*sin(D*self.alpha) - 1) - 2*cos(D*self.alpha) + 2)) + (4*self.Dm**4*(self.Dm**2*(D - 1)**2 + 2*(D - self.Dm)**2*(-D*self.alpha*sin(D*self.alpha) + self.alpha*sin(D*self.alpha) - cos(D*self.alpha) + 1)) - (2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2*(3*D**2*self.beta + 2*D*self.Dm*self.beta_1 + self.Dm**2))*(2*D*self.alpha*sin(D*self.alpha) - self.Dm*(2*D - 2*self.Dm + 2*self.alpha*sin(D*self.alpha) - 1) - 2*cos(D*self.alpha) + 2))
        den = (self.Dm**2*self.gamma*(2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**3)
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

    def get_Value(self,D):
        num = (8*D*self.Dm**3*self.alpha*(2 - D)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta*(-D**2 + 2*D - 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(D**2 - 2*D + 1) + self.Dm*(-D**2 + 2*D - 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        den = (2*self.Dm*self.gamma*(D**2 - 2*D + 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        return num/den
    
    def get_First_derivative(self,D):
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

    def get_Value(self,D):
        num = (4*self.Dm**3*(2*self.alpha*(D - self.Dm)**2 + (D - 1)**2*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - (2*D*self.beta + self.Dm)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        den = (2*self.Dm*self.gamma*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
        return num/den        
            
    def get_First_derivative(self,D):
        num = (-(4*self.Dm**3*(2*self.alpha*(D - self.Dm)**2 + (D - 1)**2*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - (2*D*self.beta_1 + self.Dm)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2)*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1)) + (2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(4*self.Dm**3*(-2*self.alpha*(D - 1)**2*(D - self.Dm) + 2*self.alpha*(D - self.Dm) + (D - 1)*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - self.beta_1*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 - (2*D*self.beta_1 + self.Dm)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1))))
        den = (self.Dm*self.gamma*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**3)
        return num/den
    
################################################################

class Functions_4_terms:
    def __init__(self,parameters):
        self.damage_function = parameters.damage_function
        
        if self.damage_function == 'cos_sin':
            self.gd_cohesive = gd_cohesive_std()
            self.hd_cohesive = hd_cohesive_cubic_4_terms(parameters)
            self.GD_bulk = GD_bulk_cos_sin(parameters)
            self.HD_bulk = HD_bulk_cos_sin_4_terms(parameters)

        elif self.damage_function == 'D_squared':
            self.gd_cohesive = gd_cohesive_std()
            self.hd_cohesive = hd_cohesive_quad_4_terms(parameters)
            self.GD_bulk = GD_bulk_D_squared(parameters)
            self.HD_bulk = HD_bulk_D_squared_4_terms(parameters)

        elif self.damage_function == 'D_std':
            self.gd_cohesive = gd_cohesive_std()
            self.hd_cohesive = hd_cohesive_quad_4_terms(parameters)
            self.GD_bulk = GD_bulk_D(parameters)
            self.HD_bulk = HD_bulk_D_4_terms(parameters)

################################################################

class Functions_3_terms:
    def __init__(self,parameters):
        self.damage_function = parameters.damage_function
    
        if self.damage_function == 'cos_sin_D_squared':
            self.gd_cohesive = gd_cohesive_twice_std()
            self.hd_cohesive = hd_cohesive_cos_D_squared_3_terms(parameters)
            self.GD_bulk = GD_bulk_cos_D_squared(parameters)

################################################################
"""
h(d) for the CZM model
"""
class hd_cohesive_czm():
    def __init__(self):
        return

    def get_Value(self,d):
        return d
    
    def get_First_derivative(self,d):
       return 1

class GD_bulk_czm():
    def __init__(self):
        return

    def get_Value(self,D):
        return np.ones_like(D)
    
    def get_First_derivative(self,D):
       return np.zeros_like(D)

class Functions_CZM:

    def __init__(self,parameters):
        self.damage_function = parameters.damage_function
    
        if self.damage_function == 'CZM':
            self.gd_cohesive = gd_cohesive_std()
            self.hd_cohesive = hd_cohesive_czm()
            self.GD_bulk = GD_bulk_czm()
            
################################################################
"""
G(D) for the LIP model
"""
class GD_bulk_lip():
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_Value(self,D):
        num = (1.-D)**2 
        den =((1.-D)**2 + (self.alpha*np.sin(self.alpha*D)*(1.-D)+1.-np.cos(self.alpha*D))*(1/self.gamma))
        return num/den
    
    def get_First_derivative(self,D):
        num = self.gamma*((D - 1.0)**2*(self.alpha **2*(D - 1.0)*cos(D*self.alpha ) + self.gamma*(2.0 - 2*D)) + (2*D - 2.0)*(-self.alpha *(D - 1.0)*sin(D*self.alpha ) + self.gamma*(D - 1.0)**2 - cos(D*self.alpha ) + 1.0))
        den = (-self.alpha *(D - 1.0)*sin(D*self.alpha ) + self.gamma*(D - 1.0)**2 - cos(D*self.alpha ) + 1.0)**2
        return num/den

"""
H(D) for the LIP model
"""
class HD_bulk_lip():
    def __init__(self,parameters):
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma

    def get_Value(self,D):
        num = (1/self.gamma)*(self.alpha*np.sin(self.alpha*D)*(1.-D)+(1.-np.cos(self.alpha*D))) 
        den = ((1.-D)+1-np.cos(self.alpha*D))**2
        return num/den
    
    def get_First_derivative(self,D):
        num = -(0.25*self.alpha**2*(D - 1.0)*(0.5*D + 0.5*cos(D*self.alpha) - 1)*cos(D*self.alpha) + 0.25*(self.alpha*sin(D*self.alpha) - 1)*(self.alpha*(D - 1.0)*sin(D*self.alpha) + cos(D*self.alpha) - 1.0))
        den = (self.gamma*(0.5*D + 0.5*cos(D*self.alpha) - 1)**3)
        return num/den

class Functions_Lip:

    def __init__(self,parameters):
        self.damage_function = parameters.damage_function

        if self.damage_function == 'LIP':
            self.GD_bulk = GD_bulk_lip(parameters)
            self.HD_bulk = HD_bulk_lip(parameters)