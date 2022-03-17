import numpy as np
from PDE_Girkmann import Girkmann
import pdb


class Girk_curve():
    '''
    Girmmann problem's curve boundary
    '''
    def __init__(self,mesh_tol=1e-12):
        PDE = Girkmann()

        self.rho_0 = PDE.rho_0
        self.theta = PDE.alpha #对应于alpha
        self.r_0 = PDE.r_0
        self.d = PDE.d
        self.a = PDE.a
        self.b = PDE.b

        self.PDE = PDE
        self.mesh_tol = mesh_tol

    def phi(self,p,xi):
        #p.shape = (NEbd,2,2)
        #xi.shape = (NQ,) or (NQ,NEbd)

        #pdb.set_trace()
        s=self.Get_theta(p) #(NEbd,2)
        
        idx, = np.nonzero((np.mean(s,axis=-1)>self.mesh_tol)&(np.mean(s,axis=-1)<self.theta-self.mesh_tol)) #只有(0,theta)的需要做曲边变换

        if len(xi.shape)>1:
            #shape = xi.shape[:-1] + p.shape[:-2] + (2,) #(NQ,NEbd,2)
            s = np.einsum('...i,i->...i',xi,s[...,0])+np.einsum('...i,i->...i',1-xi,s[...,1]) #(NQ,NEbd)
            pp = np.einsum('...i,ij->...ij',xi,p[...,0,:])+np.einsum('...i,ij->...ij',1-xi,p[...,1,:]) #(NQ,NEbd,2)
        else:
            #shape = xi.shape + p.shape[:-2] + (2,) #(NQ,NEbd,2)
            s = np.einsum('...,i->...i',xi,s[...,0])+np.einsum('...,i->...i',1-xi,s[...,1]) #(NQ,NEbd)
            pp = np.einsum('...,ij->...ij',xi,p[...,0,:])+np.einsum('...,ij->...ij',1-xi,p[...,1,:]) #(NQ,NEbd,2)
        

        if len(idx) > 0:
            pp[...,idx,0] = np.sin(s[...,idx])
            pp[...,idx,1] = np.cos(s[...,idx])
        

        r = np.mean(self.Get_r(p),axis=-1) #(NEbd,)
        idx, = np.nonzero(np.abs(r-(self.r_0-self.d/2))<self.mesh_tol)

        pp[...,idx,:] = pp[...,idx,:]*(self.r_0-self.d/2)



        idx, = np.nonzero(np.abs(r-(self.r_0+self.d/2))<self.mesh_tol)

        pp[...,idx,:] = pp[...,idx,:]*(self.r_0+self.d/2)

        return pp




    def D_xi_phi(self,p,xi):
        #p.shape = (NEbd,2,2)
        #xi.shape = (NQ,) or (NQ,NEbd)

        s=self.Get_theta(p) #(NEbd,2)

        idx, = np.nonzero((np.mean(s,axis=-1)>self.mesh_tol)&(np.mean(s,axis=-1)<self.theta-self.mesh_tol)) #只有(0,theta)的需要做曲边变换


        if len(xi.shape)>1:
            shape = xi.shape[:-1] + p.shape[:-2] + (2,) #(NQ,NEbd,2)
            s_xi = np.einsum('...i,i->...i',xi,s[...,0])+np.einsum('...i,i->...i',1-xi,s[...,1]) #(NQ,NEbd)
            
        else:
            shape = xi.shape + p.shape[:-2] + (2,) #(NQ,NEbd,2)
            s_xi = np.einsum('...,i->...i',xi,s[...,0])+np.einsum('...,i->...i',1-xi,s[...,1]) #(NQ,NEbd)



        pp = np.zeros(shape,dtype=p.dtype)
        pp[...,:,:] = (p[:,0]-p[:,1])[None,...] #


        if len(idx) > 0:
            pp[...,idx,0] = np.cos(s_xi[...,idx])*(s[...,idx,0]-s[...,idx,1])
            pp[...,idx,1] = -np.sin(s_xi[...,idx])*(s[...,idx,0]-s[...,idx,1])



        r = np.mean(self.Get_r(p),axis=-1) #(NEbd,)


        idx, = np.nonzero(np.abs(r-(self.r_0-self.d/2))<self.mesh_tol)

        pp[...,idx,:] = pp[...,idx,:]*(self.r_0-self.d/2)


        idx, = np.nonzero(np.abs(r-(self.r_0+self.d/2))<self.mesh_tol)

        pp[...,idx,:] = pp[...,idx,:]*(self.r_0+self.d/2)

        return pp










    def unit_normal(self,p):
        #求给定边界点的单位外法向, p.shape = (NQ,2)
        n = np.zeros(p.shape,dtype=p.dtype)

        s=self.Get_theta(p) #(NQ,)
        

        ####################表示该点位于曲边上#############################
        idx, = np.nonzero((s>self.mesh_tol)&(s<self.theta-self.mesh_tol))
        
        if len(idx) > 0:
            #r = self.Get_r(p[idx])
            n[idx,0] = np.sin(s[idx])
            n[idx,1] = np.cos(s[idx])

            r = self.Get_r(p[idx])
            idx_temp, = np.nonzero(np.abs(r-(self.r_0-self.d/2))<self.mesh_tol)

            n[idx[idx_temp]] = -n[idx[idx_temp]] 


        ####################表示该点位于x=0的直边上########################
        idx, = np.nonzero(np.abs(s)<=self.mesh_tol)

        if len(idx) > 0:
            n[idx,0] = -1.0
            n[idx,1] = -0.0



        ############################右边小矩形区域#########################
        idx, = np.nonzero(s>=self.theta-self.mesh_tol)

        if len(idx) > 0:
            x_left = (self.r_0 - self.d/2)*np.sin(self.theta)
            x_right = x_left + self.a

            y_above = (self.r_0 + self.d/2)*np.cos(self.theta)
            y_below = y_above - self.b

            idx_temp, = np.nonzero(np.abs(p[idx,0]-x_left)<self.mesh_tol)
            n[idx[idx_temp],0] = -1.0
            
            idx_temp, = np.nonzero(np.abs(p[idx,0]-x_right)<self.mesh_tol)
            n[idx[idx_temp],0] = 1.0

            idx_temp, = np.nonzero(np.abs(p[idx,1]-y_above)<self.mesh_tol)
            n[idx[idx_temp],1] = 1.0

            idx_temp, = np.nonzero(np.abs(p[idx,1]-y_below)<self.mesh_tol)
            n[idx[idx_temp],1] = -1.0



        return n

    
    def Get_theta(self,p):
        '''
        对应于alpha
        '''
        return self.PDE.get_alpha(p)
    
    def Get_r(self,p):
        '''
        得到极坐标r
        '''
        return self.PDE.get_r(p) 
