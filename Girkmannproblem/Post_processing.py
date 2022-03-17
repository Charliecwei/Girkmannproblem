import numpy as np
from fealpy.quadrature.GaussLegendreQuadrature import GaussLegendreQuadrature





class Post():
    def __init__(self,PDE,sh,tspace):
        mesh = tspace.mesh
        bc = np.array([0.5,0.5])
        pp = mesh.bc_to_point(bc) #(NE,2)


        alpha = PDE.get_alpha(pp) #(NE,)
        index, = np.nonzero(np.abs(alpha-PDE.alpha)<1e-12) #(NE0,)
        self.index = index



        self.mesh = mesh
        self.tspace = tspace
        self.PDE = PDE
        self.sh = sh 


        self.qf = GaussLegendreQuadrature(tspace.p+5)











    def g(self,bc,index=None):
        ##算给定边的sigma*n
        index = self.index if index is None else index

        shape = bc.shape[:-1] + index.shape + (2,) #(NQ,NE,2)
        val = np.zeros(shape,dtype=np.float64) #(NQ,NE,2)


        boundary_edge_flag = self.mesh.ds.boundary_edge_flag()
        inner_edge_index, = np.nonzero(~boundary_edge_flag)
        boundary_edge_index, = np.nonzero(boundary_edge_flag)

        #############内部边#################
        idx_edge, idx_index = np.where(inner_edge_index[:,None] == index) 
        if len(idx_edge)>0:
            n = self.mesh.edge_unit_normal(bc,index=index[idx_index]) #(NQ,NEin,2)


            phi = self.tspace.edge_basis(bc,index=idx_edge) #(NQ,NEin,ldof,3,tdim)
            edge2dof = self.tspace.Inedge2dof[idx_edge] #(NEin,ldof,3)
            sh = self.sh[edge2dof] #(NEin,ldof,3)


            shape = list(phi.shape)
            shape[-1] = 2
            phin = np.zeros(shape,dtype=np.float64) #(NQ,NEin,ldof,3,gdim)
            

            phin[...,0] = np.einsum('...ijlk,...ik->...ijl',phi[...,[0,2]],n)
            phin[...,1] = np.einsum('...ijlk,...ik->...ijl',phi[...,[2,1]],n)


            val[:,idx_index] = np.einsum('ijk,...ijkl->...il',sh,phin) #(NQ,NEin,3)

        ############边界边#############################
        idx_edge, idx_index = np.where(boundary_edge_index[:,None] == index)
        if len(idx_edge)>0:
            n = self.mesh.edge_unit_normal(bc,index=index[idx_index]) #(NQ,NEbd,2)


            phi = self.tspace.edge_basis(bc,index=idx_edge,is_bd_dof=True) #(NQ,NEbd,ldof,3,tdim)
            edge2dof = self.tspace.Bdedge2dof[idx_edge] #(NEbd,ldof,3)
            sh = self.sh[edge2dof] #(NEbd,ldof,3)

            shape = list(phi.shape)
            shape[-1] = 2
            phin = np.zeros(shape,dtype=np.float64) #(NQ,NEbd,ldof,3,gdim)
            
            phin[...,0] = np.einsum('...ijlk,...ik->...ijl',phi[...,[0,2]],n)
            phin[...,1] = np.einsum('...ijlk,...ik->...ijl',phi[...,[2,1]],n)

            val[:,idx_index] = np.einsum('ijk,...ijkl->...il',sh,phin) #(NQ,NEbd,3)




        return val #(NQ,NE0,2)



    def get_Q(self):
        index = self.index
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        mesh = self.mesh

        rm = mesh.reference_cell_measure(TD=1)
        D = mesh.first_fundamental_form(bcs,index=index)
        D = np.sqrt(np.linalg.det(D))#(NQ,NEbd)


        pp = self.mesh.bc_to_point(bcs,index=index) #(NQ,NE0,2)
        W_Q = self.PDE.W_Q(pp) #(NQ,NE0,2)


        g = self.g(bcs,index=index) #(NQ,NE0,2)
        
        val = np.einsum('i,ijk,ijk,ij->',ws*rm,W_Q,g,D)/self.PDE.rho_0
        val *= 1e9
        return val


    def get_M(self):
        index = self.index
        bcs, ws = self.qf.get_quadrature_points_and_weights()
        mesh = self.mesh
        
 
        rm = mesh.reference_cell_measure(TD=1)
        D = mesh.first_fundamental_form(bcs,index=index)
        D = np.sqrt(np.linalg.det(D))#(NQ,NEbd)


        pp = self.mesh.bc_to_point(bcs,index=index) #(NQ,NE0,2)
        W_M = self.PDE.W_M(pp) #(NQ,NE0,2)

        g = self.g(bcs) #(NQ,NE0,2)


        val = np.einsum('i,ijk,ijk,ij->',ws,W_M,g,D)/self.PDE.rho_0
        val *= 1e9
        return val
