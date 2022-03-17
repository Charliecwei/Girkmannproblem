import numpy as np

class Girkmann():

    def __init__(self):
        '''
        Girkmann problem model
        '''
        #E = 2.059e10 #20.59Gpa
        E = 20.59 #Gpa
        nu = 0

        self.lam = E*nu/(1+nu)*(1-2*nu)
        self.mu = E/(2*(1+nu))

        #self.F = 32690#N 
        #self.p = 27256#Pa
        self.F = 3.2690e-5 #GN
        self.p = 2.7256e-5 #Gpa

        self.rho_0 = 15.0
        self.alpha = 2*np.pi/9
            
        self.r_0 =  self.rho_0/np.sin(self.alpha)
        self.z_0 = self.r_0*np.cos(self.alpha)


        self.a = 0.60
        self.b = 0.50
        self.d = 0.06
        



    def get_alpha(self,p):
        #得到极坐标的alpha
        shape = p.shape[:-1]
        p = p.reshape(-1,2)
        r = np.sqrt(np.sum(p**2,axis=-1))
        s = np.arccos(p[:,1]/r) #[0,pi]

        if np.sum((s>np.pi/2))>0:
            raise ValueError("some points are not right!")


        return s.reshape(shape)


    def get_r(self,p):
        #得到极坐标的r
        return np.sqrt(np.sum(p**2,axis=-1))


    def W_Q(self,p):
        r = np.sqrt(np.sum(p**2,axis=-1))
        return p/r[...,None]

    def W_M(self,p):
        z_0 = self.z_0
        rho_0 = self.rho_0
        p[...,0] = (p[...,0] - rho_0)
        p[...,1] = -(p[...,1] - z_0)
        p = p[...,[1,0]]

        return p
    

