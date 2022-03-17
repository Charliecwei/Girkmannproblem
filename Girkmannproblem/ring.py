import gmsh
import numpy as np
from fealpy.mesh.CurveLagrangeTriangleMesh import CurveLagrangeTriangleMesh
from Curve_boundary import Girk_curve
import matplotlib.pyplot as plt
from PDE_Girkmann import Girkmann

def ring(hs,h=0.5,mdegree=3):
    PDE = Girkmann()

    pi = np.pi
    sin = np.sin
    cos = np.cos

    rho_0 = PDE.rho_0
    alpha = PDE.alpha
    r_0 = PDE.r_0
    d = PDE.d
    a = PDE.a
    b = PDE.b

    #curve = Girk_curve()
    curve = ring_curve()
    ##################gmsh###############
    gmsh.initialize()

    
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha),(r_0-d/2)*cos(alpha),0,hs,1)
    gmsh.model.geo.addPoint((r_0+d/2)*sin(alpha),(r_0+d/2)*cos(alpha),0,hs,2)

    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha), (r_0+d/2)*cos(alpha)-b, 0, h, 3)
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha)+a, (r_0+d/2)*cos(alpha)-b, 0, h,4)
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha)+a, (r_0+d/2)*cos(alpha), 0, h, 5)


    gmsh.model.geo.addLine(2,1,1)
    gmsh.model.geo.addLine(1,3,2)
    gmsh.model.geo.addLine(3,4,3)
    gmsh.model.geo.addLine(4,5,4)
    gmsh.model.geo.addLine(5,2,5)
    
    gmsh.model.geo.addCurveLoop([1,2,3,4,5],1)
    gmsh.model.geo.addPlaneSurface([1], 1)



    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    node = gmsh.model.mesh.getNodes()[1]

    NN = node.shape[0]//3
    node = node.reshape(NN,3)
    node = np.array(node[:,:2],dtype=np.float64)

    cell = np.array(gmsh.model.mesh.getElements(dim=2)[2][0]-1,dtype=np.int64)
    NC = cell.shape[0]//3
    cell = cell.reshape(NC,3)


    mesh = CurveLagrangeTriangleMesh(node,cell,p=mdegree,curve=curve)


    ################gmesh################
    gmsh.finalize()

    return mesh


class ring_curve():
    '''
    Girmmann problem's ring curve boundary
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

        if len(xi.shape)>1:
            pp = np.einsum('...i,ij->...ij',xi,p[...,0,:])+np.einsum('...i,ij->...ij',1-xi,p[...,1,:]) #(NQ,NEbd,2)
        
        else:
            pp = np.einsum('...,ij->...ij',xi,p[...,0,:])+np.einsum('...,ij->...ij',1-xi,p[...,1,:]) #(NQ,NEbd,2)
            
        return pp



    def D_xi_phi(self,p,xi):
        if len(xi.shape)>1:
            shape = xi.shape[:-1] + p.shape[:-2] + (2,) #(NQ,NEbd,2)
            
        else:
            shape = xi.shape + p.shape[:-2] + (2,) #(NQ,NEbd,2)
        
        pp = np.zeros(shape,dtype=p.dtype)
        pp[...,:,:] = (p[:,0]-p[:,1])[None,...] #

        return pp



    def unit_normal(self,p):
        #求给定边界点的单位外法向, p.shape = (NQ,2)
        n = np.zeros(p.shape,dtype=p.dtype)
        s = self.PDE.get_alpha(p)
        
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

        idx, = np.nonzero(np.abs(s-self.theta)<self.mesh_tol)

        if len(idx)>0:
            n[idx,0] = -np.cos(self.theta)
            n[idx,1] = np.sin(self.theta)

        return n    



