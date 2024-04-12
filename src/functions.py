
import numpy as np
import scipy

from numpy import cos,sin


class Functions :

    def __init__(self,parameters) :

        self.alpha = parameters.alpha
        self.beta = parameters.beta
        self.beta_1 = parameters.beta_1
        self.choice = parameters.choice
        self.gamma = parameters.gamma
        self.Dm = parameters.Dm

    
    def gd_cohesive(self, d):
        """Cohesive function."""
        return (1. / d) - 1.
    
    def dgd_cohesive_dd(self, d):
        """Derivative of the cohesive function."""
        return -1. / d**2
    
    def hd_cohesive(self, d):
        """Cohesive energy dissipation function."""
        # 1-cos(aD)/(1-D)
        if self.choice == 1 :
            return self.beta * d**3 + self.beta_1 * d**2 + d
        
        # a(1-(1-d)**2)
        if self.choice == 2 :
            return  self.beta * d**2 + d
        
        #aD/(1-D)
        if self.choice == 3 :
            return  self.beta * d**2 + d
        
        else : 
            return print("Choice of Choice is wrong")
        
    def dhd_cohesive_dd(self, d):
        """Derivative of the cohesive energy dissipation function."""
        # 1-cos(aD)/(1-D)
        if self.choice == 1 :
            return 3 * self.beta * d**2 + 2 * self.beta_1 * d + 1
        
        # a(1-(1-D)**2)/(1-D)
        if self.choice == 2 :
            return 2 * self.beta * d  + 1

        #aD/(1-D)
        if self.choice == 3 :
            return 2 * self.beta * d  + 1
        
        else : 
            return print("Choice of Choice is wrong")
    
    def GD_bulk(self, D):
        """Bulk softening function."""
        # 1-cos(aD)/(1-D)
        if self.choice == 1 :
            return (1 - D)**2 / ((1 - D)**2 + (self.alpha * sin(self.alpha * D) * (1 - D) + 1 - cos(self.alpha * D)) / (self.gamma))
        
        # a(1-(1-D)**2)/(1-D)
        if self.choice == 2:
            return ((1-D)**2)/((1-D)**2 + ((self.alpha*(-D*(D-2)+2*(D-1)**2)-(2*self.alpha*(1-D)**2)))/(self.gamma))

        #aD/(1-D)
        if self.choice == 3 :        
            return ((1-D)**2)/((1-D)**2 + (self.alpha-self.alpha*(1-D)**2)/(self.gamma))
        
        else : 
            return print("Choice of Choice is wrong")
        
    def dGD_bulk_dD(self, D):
        """Derivative of the bulk softening function."""
        # 1-cos(aD)/(1-D)
        if self.choice == 1 :
           num =  self.gamma*(D - 1)*(-2*self.alpha*(D - 1)*sin(D*self.alpha) + 2*self.gamma*(D - 1)**2 + (D - 1)*(self.alpha**2*(D - 1)*cos(D*self.alpha) + 2*self.gamma*(1 - D)) - 2*cos(D*self.alpha) + 2)
           den = (-self.alpha*(D - 1)*sin(D*self.alpha) + self.gamma*(D - 1)**2 - cos(D*self.alpha) + 1)**2
           return num/den
        
        # a(1-(1-D)**2)/(1-D)
        if self.choice == 2:
            num = 2*self.gamma*(D - 1)*(-2*self.alpha*(D - 1)**2 - self.alpha*(D*(D - 2) - 2*(D - 1)**2) + self.gamma*(D - 1)**2 + (D - 1)*(self.alpha*(D - 1) + self.gamma*(1 - D)))
            den = (-2*self.alpha*(D - 1)**2 - self.alpha*(D*(D - 2) - 2*(D - 1)**2) + self.gamma*(D - 1)**2)**2
            return num/den
        
        #aD/(1-D)
        if self.choice == 3 :        
            num =2*self.gamma*(D - 1)*(-self.alpha*(D - 1)**2 + self.alpha + self.gamma*(D - 1)**2 + (D - 1)*(self.alpha*(D - 1) + self.gamma*(1 - D)))
            den = (-self.alpha*(D - 1)**2 + self.alpha + self.gamma*(D - 1)**2)**2
            return num/den
        
        else : 
            return print("Choice of Choice is wrong")
        
    def HD_bulk(self, D):
        """Derivative of the bulk energy dissipation function."""
        # 1-cos(aD)/(1-D)
        if self.choice == 1:
            num = 4*self.Dm**4*(self.Dm**2*(D - 1)**2 + 2*(D - self.Dm)**2*(-D*self.alpha*sin(D*self.alpha) + self.alpha*sin(D*self.alpha) - cos(D*self.alpha) + 1)) - (2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2*(3*D**2*self.beta + 2*D*self.Dm*self.beta_1 + self.Dm**2)
            den = (2*self.Dm**2*self.gamma*(2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2)
            return num/den
        
        # a(1-(1-D)**2)/(1-D)
        if self.choice == 2:
            num = (8*D*self.Dm**3*self.alpha*(2 - D)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta*(-D**2 + 2*D - 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(D**2 - 2*D + 1) + self.Dm*(-D**2 + 2*D - 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
            den = (2*self.Dm*self.gamma*(D**2 - 2*D + 1)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
            return num/den
        
        #aD/(1-D)
        if self.choice == 3 : 
            num = (4*self.Dm**3*(2*self.alpha*(D - self.Dm)**2 + (D - 1)**2*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - (2*D*self.beta + self.Dm)*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
            den = (2*self.Dm*self.gamma*(2*D**3*self.alpha + D*self.Dm*(-2*D*self.alpha - D + 1) + 2*self.Dm**2*(D - 1))**2)
            return num/den        
            
        else : 
            return print("Choice of Choice is wrong")
        
    
    def dHD_bulk_dD(self, D):
        """Derivative of the bulk energy dissipation function."""
        # 1-cos(aD)/(1-D)
        if self.choice == 1:
            num = ((2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))*(4*self.Dm**4*(self.Dm**2*(D - 1) + self.alpha**2*(1 - D)*(D - self.Dm)**2*cos(D*self.alpha) + 2*(-D + self.Dm)*(D*self.alpha*sin(D*self.alpha) - self.alpha*sin(D*self.alpha) + cos(D*self.alpha) - 1)) - (3*D*self.beta + self.Dm*self.beta_1)*(2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2 + (2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))*(3*D**2*self.beta + 2*D*self.Dm*self.beta_1 + self.Dm**2)*(2*D*self.alpha*sin(D*self.alpha) - self.Dm*(2*D - 2*self.Dm + 2*self.alpha*sin(D*self.alpha) - 1) - 2*cos(D*self.alpha) + 2)) + (4*self.Dm**4*(self.Dm**2*(D - 1)**2 + 2*(D - self.Dm)**2*(-D*self.alpha*sin(D*self.alpha) + self.alpha*sin(D*self.alpha) - cos(D*self.alpha) + 1)) - (2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**2*(3*D**2*self.beta + 2*D*self.Dm*self.beta_1 + self.Dm**2))*(2*D*self.alpha*sin(D*self.alpha) - self.Dm*(2*D - 2*self.Dm + 2*self.alpha*sin(D*self.alpha) - 1) - 2*cos(D*self.alpha) + 2))
            den = (self.Dm**2*self.gamma*(2*D*(cos(D*self.alpha) - 1) + self.Dm*((D - 1)*(D - 2*self.Dm) - 2*cos(D*self.alpha) + 2))**3)
            return num/den
        
        # a(1-(1-D)**2)/(1-D)
        if self.choice == 2:
            num = ((D - 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(-D**2 + 2*D - 1) + self.Dm*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2) + (D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(-8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm) - 8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)*(D - self.Dm)**2 - 4*D*self.Dm**3*self.alpha*(D - 1)**2*(D - self.Dm)**2 - 2*D*self.beta_1*(D - 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 - 2*D*self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1)) + 4*self.Dm**5*(D - 1)**3 + 4*self.Dm**5*(D - 1)*(D**2 - 2*D + 1) - 4*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm)**2 - self.Dm*(D - 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 - self.Dm*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1)) - self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2) + (D**2 - 2*D + 1)*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1))*(8*D*self.Dm**3*self.alpha*(D - 2)*(D - 1)**2*(D - self.Dm)**2 + 2*D*self.beta_1*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 + 4*self.Dm**5*(D - 1)**2*(-D**2 + 2*D - 1) + self.Dm*(D**2 - 2*D + 1)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2))
            den = (self.Dm*self.gamma*(D**2 - 2*D + 1)**2*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**3)
            return num/den

        #aD/(1-D)
        if self.choice == 3 :
            num = (-(4*self.Dm**3*(2*self.alpha*(D - self.Dm)**2 + (D - 1)**2*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - (2*D*self.beta_1 + self.Dm)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2)*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1)) + (2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(4*self.Dm**3*(-2*self.alpha*(D - 1)**2*(D - self.Dm) + 2*self.alpha*(D - self.Dm) + (D - 1)*(self.Dm**2 - 2*self.alpha*(D - self.Dm)**2)) - self.beta_1*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**2 - (2*D*self.beta_1 + self.Dm)*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))*(6*D**2*self.alpha - D*self.Dm*(2*self.alpha + 1) + 2*self.Dm**2 - self.Dm*(2*D*self.alpha + D - 1))))
            den = (self.Dm*self.gamma*(2*D**3*self.alpha - D*self.Dm*(2*D*self.alpha + D - 1) + 2*self.Dm**2*(D - 1))**3)
            return num/den

        else : 
            return print("Choice of Choice is wrong")
        

