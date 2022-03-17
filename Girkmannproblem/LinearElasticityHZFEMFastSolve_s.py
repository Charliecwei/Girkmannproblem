###############################胡张元程序快速求解算法############################
import numpy as np
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, cg
from timeit import default_timer as dtimer
from scipy.sparse import spdiags, tril, triu, csr_matrix, bmat, construct
from fealpy.decorator import timer
from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace
import transplant
from fealpy.solver import MatlabSolver
import pdb

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))



class GaussSeidelSmoother():
    def __init__(self, A):
        self.L0 = tril(A).tocsr()
        self.U0 = triu(A, k=1).tocsr()

        self.U1 = self.L0.T.tocsr()
        self.L1 = self.U0.T.tocsr()        
        
    def smooth(self, b, x0, lower=True, maxit=3):
        if lower:
            for i in range(maxit):
                x0[:] = spsolve(self.L0, b-self.U0@x0, permc_spec="NATURAL")
        else:
            for i in range(maxit):
                x0[:] = spsolve(self.U1, b-self.L1@x0, permc_spec="NATURAL")




class Symmetric_GaussSeidelSmoother():
    def __init__(self,A):
        self.L = tril(A).tocsr()
        self.U = triu(A).tocsr()
        self.D = A.diagonal()
        self.A = A

    def smooth(self, b, x0, maxit=3):
        for i in range(maxit):
            x0[:] = x0 + spsolve(self.U,self.D*spsolve(self.L,b-self.A@x0))



class JacobiSmoother():
    def __init__(self,A):
        self.D = A.diagonal()
        self.L = tril(A, k=-1).tocsr()
        self.U = triu(A, k=1).tocsr()


    def smooth(self, b, x0, maxit=3):
       for i in range(maxit):
           x0[:] = b - self.L@x0 - self.U@x0
           x0 /= self.D












class LinearElasticityHZFEMFastSolve():
    def __init__(self,A,F,vspace,tspace,mu,lam,isBDdof=None):
        '''

        Notes
        -----
            求解胡张元形成的线弹性力学方程
            A = (M, B, C)
            F = (F0,F1) 

            离散的代数系统如下
            M x0 + B^T x1 = F0
            B x0 - C x1   = F1

        '''
        self.M = A[0]
        self.B = A[1]
        self.C = A[2]
        self.vspace = vspace
        self.mu = mu
        
        x0 = np.zeros_like(F[0])
        x1 = np.zeros_like(F[1].T.reshape(-1))

        if lam == 0:
            self.D = self.M.diagonal()
        else:
            self.D = tspace.compliance_tensor_matrix(mu=mu,lam=0.0).diagonal()


        
        if isBDdof is not None:
            isfreedof, = np.nonzero(~isBDdof)
            isBDdof, = np.nonzero(isBDdof)
            x0[isBDdof] = F[0][isBDdof]

            self.D = self.D[isfreedof]
            self.M = self.M[isfreedof][:,isfreedof]
            self.B = self.B[:,isfreedof]
            self.isfreedof = isfreedof
         
        else:
            self.isfreedof = np.arange(self.M.shape[0])




        self.tgdof = self.M.shape[0]
        self.vgdof = self.B.shape[0]
        # S 相当于间断元的刚度矩阵
        self.S = self.B@spdiags(1/self.D,0,self.tgdof,self.tgdof)@self.B.T
        if self.C is not None:
            self.S += self.C


        self.smoother = GaussSeidelSmoother(self.S)

        self.iter = 0


        self.x = np.r_[x0,x1]
        self.F = np.r_[F[0][self.isfreedof],F[1].T.reshape(-1)]
        self.isfreedof = np.r_[self.isfreedof,A[0].shape[0]+np.arange(A[1].shape[0])]

        print(np.linalg.cond(self.S.todense()))
        s
        self.init_construct_amg_solver()
        
         
    def init_construct_amg_solver(self):

        # construct amg solver
        vspace = self.vspace
        mesh = vspace.mesh
        cspace = LagrangeFiniteElementSpace(mesh,1)

        # Get interpolation matrix
        NC = mesh.number_of_cells()
        bc = vspace.dof.multiIndex/vspace.p #(fldof,gdim+1)
        val = np.tile(bc, (NC, 1)) #(NC*fldof,gdim+1)

        gdim = mesh.geo_dimension()
        self.gdim = gdim
        fldof = vspace.number_of_local_dofs() #f表示细空间
        cldof = cspace.number_of_local_dofs() #c表示粗空间
        fgdof = vspace.number_of_global_dofs()
        cgdof = cspace.number_of_global_dofs()

        I = np.broadcast_to(vspace.cell_to_dof()[:,:,None],shape=(NC,fldof,gdim+1))
        J = np.broadcast_to(cspace.cell_to_dof()[:,None,:],shape=(NC,fldof,cldof))

        self.PI = csr_matrix((val.flat, (I.flat, J.flat)), shape=(fgdof, cgdof))
        
       

        is_free_dof, = np.nonzero(~cspace.is_boundary_dof())
       

        #########################coarse slover#######################
        self.PI = self.PI[:,is_free_dof]

        self.PI = bmat([[self.PI,None],[None,self.PI]],format='csr')
        self.S_coarse1 = self.PI.T@self.S@self.PI
        
        
        is_free_dof = np.r_[is_free_dof,is_free_dof+cspace.number_of_global_dofs()]
        self.S_coarse2 = cspace.linear_elasticity_matrix(0,self.mu)[is_free_dof][:,is_free_dof]
        
        





    def linear_operator(self,b):
        m = self.tgdof
        r = np.zeros_like(b)
        
        r[:m] = self.M@b[:m]+self.B.T@b[m:]
        r[m:] = self.B@b[:m]
        if self.C is not None:
              r[m:] -= self.C@b[m:]

        return r



    def precondieitoner(self,r):
        tgdof = self.tgdof
        u0 = r[:tgdof]/self.D
        r1 = r[tgdof:]-self.B@u0

        if False:
             u1 = spsolve(self.S,r1)

        elif True:

             u1 = np.zeros_like(r1)
             self.smoother.smooth(r1,u1,maxit=3)
             r2 = r1 - self.S@u1
            
             if False:
                 u1 = u1
             elif True:
                 u1 += self.PI@spsolve(self.S_coarse1,self.PI.T@r2)

             elif True:
                 u1 += self.PI@spsolve(self.S_coarse2,self.PI.T@r2)



             self.smoother.smooth(r1,u1,maxit=3)
             #self.smoother.smooth(r1,u1,lower=False,maxit=10)

        ####统计调用次数###
        self.iter += 1
        print(self.iter)
        ##################
        return np.r_[u0+self.B.T@u1[:self.vgdof]/self.D, -u1]
    


    @timer
    def solve(self, tol=1e-8):
        m = self.tgdof
        n = self.vgdof
        gdof = m + n
        counter = IterationCounter(disp=False)
        F = np.copy(self.F)
        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.precondieitoner)
        self.x[self.isfreedof], info = gmres(A, F,M=P, tol=tol, callback=counter)
        #x, info = lgmres(A, F, tol=1e-14, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)
        return self.x 












class LinearElasticityHZFEMFastSolve_pure_stress():
    def __init__(self,A,F,vspace,tspace,mu,lam,isBDdof):
        '''

        Notes
        -----
            求解胡张元形成的线弹性力学方程,纯应力边界，差一个刚性位移
            加条件:
              \int_{\Omgea} u dxdy = 0
              \int_{\Omega} v dxdy = 0
              \int_{\Omgea} (u_y-v_x) dxdy = 0


            离散代数系统如下:
                 M x0 + B^T x1         = F0
                 B x0 - C x1 + C0^T x2 = F1
                        C0 x1          = 0
            
       '''     
        self.M = A[0]
        self.B = A[1]
        self.C = A[2]
        self.vspace = vspace
        self.mu = mu
        #matlab = transplant.Matlab(executable='/Applications/MATLAB_R2021a.app/bin/matlab')
        #self.solver = MatlabSolver(matlab)


        x0 = np.zeros_like(F[0])

        if lam == 0:
            self.D = self.M.diagonal()
        else:
            self.D = tspace.compliance_tensor_matrix(mu=mu,lam=0.0).diagonal()


        isfreedof, = np.nonzero(~isBDdof)
        isBDdof, = np.nonzero(isBDdof)
        x0[isBDdof] = F[0][isBDdof]

        self.D = self.D[isfreedof]
        self.M = self.M[isfreedof][:,isfreedof]
        self.B = self.B[:,isfreedof]
        self.isfreedof = isfreedof
        
        #print(np.max(vspace.cellmeasure)/np.min(vspace.cellmeasure)) 
        #print(np.max(self.D)/np.min(self.D))
        
        #ss 
        #pdb.set_trace()
        #对位移加限制条件，抹去刚性位移
        if False:
            self.Pu = csr_matrix(np.eye(A[1].shape[0]))
        elif False:
            isvfreedof = np.ones(A[1].shape[0],dtype=bool)
            n = A[1].shape[0]//2
            #ldof = vspace.number_of_local_dofs(is_bd_dof=True)
            idx = vspace.Bdcell2dof[0,-1]
            isvfreedof[n+idx] = False
            isvfreedof, = np.nonzero(isvfreedof)
            self.Pu = csr_matrix(np.eye(A[1].shape[0])[:,isvfreedof])
        elif False:
            #################construct C0################
            c = vspace.integral_basis()#积分
            n = c.shape[0] 

            Q,R=np.linalg.qr(c[:,None],mode='complete')

            idx = np.nonzero((np.sum(np.abs(R),axis=-1) < 1e-14))[0]
            self.Pu = Q[idx].T
            idx = np.abs(self.Pu)<1e-14
            self.Pu[idx]=0
            #self.Pu = csr_matrix(self.Pu)
            self.Pu = bmat([[np.eye(n),None],[None,self.Pu]],format='csr')
        elif True:
            c = vspace.integral_basis()#积分
            n = c.shape[0]

            idx = np.abs(c)<1e-14
            c[idx] = 0
            
            idx, = np.nonzero(~idx)
            idx = idx[0]
            Pu = np.eye(n)
            Pu[idx] = -c/c[idx]
            
            Pu = np.delete(Pu,idx,axis=-1)#(n,n-1) 
            Pu = bmat([[np.eye(n),None],[None,Pu]],format='csr')
            
            self.Pu = Pu
            

            
            

        #print(np.linalg.matrix_rank(self.B.todense()))
        print(self.Pu.nnz)
        print(self.B.nnz)
        self.B = self.Pu.T@self.B
        print(self.B.nnz,self.B.shape) 
        #print(np.linalg.matrix_rank(self.B.todense()))


        self.tgdof = self.M.shape[0]
        self.vgdof = self.B.shape[0]
        # S 相当于间断元的刚度矩阵
        self.S = self.B@spdiags(1/self.D,0,self.tgdof,self.tgdof)@self.B.T
        if self.C is not None:
            self.C = self.Pu.T@self.C@self.Pu
            self.S += self.C

        F1 = self.Pu.T@F[1].T.reshape(-1)
        x1 = np.zeros_like(F1)
        self.x = np.r_[x0,x1]
        self.F = np.r_[F[0][self.isfreedof],F1]
        self.isfreedof = np.r_[self.isfreedof,A[0].shape[0]+np.arange(self.vgdof)]



        self.iter = 0
        self.smoother = GaussSeidelSmoother(self.S)
        
        self.init_construct_amg_solver()
        print(np.min(self.D),np.max(self.D))
        print(np.max(np.abs(self.S)))
        print(np.linalg.matrix_rank(self.S.todense()),self.S.shape)
        print(np.linalg.cond(self.S.todense()))
        ss         
        
        #pdb.set_trace()


    def linear_operator(self,b):
        m = self.tgdof
        n = self.vgdof
        r = np.zeros_like(b)


        r[:m] = self.M@b[:m]+self.B.T@b[m:m+n]
        r[m:m+n] = self.B@b[:m]

        if self.C is not None:
            r[m:m+n] -= self.C@b[m:m+n]

        return r





    def precondieitoner(self,r):
        m = self.tgdof
        n = self.vgdof
        
        u0 = r[:m]/self.D
        r1 = r[m:] - np.r_[self.B@u0]
        #print(self.S.shape) 
        if True:
            #u1 = spsolve(self.S,r1)
            u1 = self.solver.divide(self.S,r1)
        elif True:
            u1 = np.zeros_like(r1)
            self.smoother.smooth(r1,u1,maxit=10)
            r2 = r1 - self.S@u1
            
            if False:
                u1 = u1
            elif True:
                u1 += self.PI@spsolve(self.S_coarse1,self.PI.T@r2)
                #u1 += self.PI@self.solver.divide(self.S_coarse1,self.PI.T@r2)
            elif True:
                u1 += self.PI@spsolve(self.S_coarse2,self.PI.T@r2)
            
            self.smoother.smooth(r1,u1,lower=False,maxit=10)



        ####统计调用次数####
        self.iter += 1
        print(self.iter)
        return np.r_[u0+self.B.T@u1[:n]/self.D, -u1]







    def init_construct_amg_solver(self):
        vspace = self.vspace
        mesh = vspace.mesh
        # Get interpolation matrix
        if 'NCin' in dir(vspace):
            #内部单元
            boundary_cell_flag = mesh.ds.boundary_cell_flag()
            inner_cell_index, = np.nonzero(~boundary_cell_flag)
            boundary_cell_index, = np.nonzero(boundary_cell_flag)

            cspace = LagrangeFiniteElementSpace(mesh,1)
            NC = vspace.number_of_inner_cell()
            bc = vspace.inner_cell_multiIndex/vspace.inner_p
            val0 = np.tile(bc, (NC, 1))
            
            fldof = vspace.number_of_local_dofs() #f表示细空间
            cldof = cspace.number_of_local_dofs() #c表示粗空间
            fgdof = vspace.number_of_global_dofs()
            cgdof = cspace.number_of_global_dofs()
           
            I0 = np.broadcast_to(vspace.Incell2dof[:,:,None],shape=(NC,fldof,cldof))
            J0 = np.broadcast_to(cspace.cell_to_dof()[inner_cell_index,None,:],shape=(NC,fldof,cldof))

            

            #边界单元
            NC = vspace.number_of_boundary_cell()
            bc = vspace.boundary_cell_multiIndex/vspace.boundary_p
            val1 = np.tile(bc, (NC, 1))

            fldof = vspace.number_of_local_dofs(is_bd_dof=True)

            I1 = np.broadcast_to(vspace.Bdcell2dof[:,:,None],shape=(NC,fldof,cldof))
            J1 = np.broadcast_to(cspace.cell_to_dof()[boundary_cell_index,None,:],shape=(NC,fldof,cldof))
            
            val = np.r_[val0.flat,val1.flat]
            I = np.r_[I0.flat,I1.flat]
            J = np.r_[J0.flat,J1.flat]
            
            self.PI = csr_matrix((val, (I, J)), shape=(fgdof, cgdof))

        else:        
            cspace = LagrangeFiniteElementSpace(mesh,1)
            NC = mesh.number_of_cells()
            bc = vspace.dof.multiIndex/vspace.p #(fldof,gdim+1)
            val = np.tile(bc, (NC, 1)) #(NC*fldof,gdim+1)

            gdim = mesh.geo_dimension()
            self.gdim = gdim
            fldof = vspace.number_of_local_dofs() #f表示细空间
            cldof = cspace.number_of_local_dofs() #c表示粗空间
            fgdof = vspace.number_of_global_dofs()
            cgdof = cspace.number_of_global_dofs()

            I = np.broadcast_to(vspace.cell_to_dof()[:,:,None],shape=(NC,fldof,gdim+1))
            J = np.broadcast_to(cspace.cell_to_dof()[:,None,:],shape=(NC,fldof,cldof))

            self.PI = csr_matrix((val.flat, (I.flat, J.flat)), shape=(fgdof, cgdof))



        is_free_dof, = np.nonzero(~cspace.is_boundary_dof())

        #########################coarse slover#######################
        self.PI = self.PI[:,is_free_dof]
        
        c = vspace.integral_basis()[is_free_dof]#积分
        idx = np.abs(c) < 1e-14
        c[idx] = 0
        n = c.shape[0]

        idx, = np.nonzero(c!=0)
        if len(idx) > 0: 
            idx = idx[0]
            Pu = np.eye(n)
            Pu[idx] = -c/c[idx]
            Pu = np.delete(Pu,idx,axis=-1)#(n,n-1)
            
            self.PI = self.PI@Pu
             


        self.PI = bmat([[self.PI, None],
                        [None, self.PI]],format='csr')
        

        self.PI = self.Pu.T@self.PI
        self.S_coarse1 = self.PI.T@self.S@self.PI
    
        
        #S_coarse = cspace.linear_elasticity_matrix(0,self.mu)
        #is_free_dof = np.r_[is_free_dof,is_free_dof+cspace.number_of_global_dofs()]
        #self.S_coarse2 = S_coarse[is_free_dof][:,is_free_dof]
        #print(np.linalg.cond(self.S_coarse1.todense()))
        #print(np.linalg.matrix_rank(self.S_coarse1.todense()),self.S_coarse1.shape)
        #print(self.S_coarse1.todense())
    @timer
    def solve(self, tol=1e-8):
        m = self.tgdof
        n = self.vgdof
        gdof = m+n

        counter = IterationCounter(disp=False)
        F = np.copy(self.F)
        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.precondieitoner)
        self.x[self.isfreedof], info = gmres(A, F,M=P, tol=tol, callback=counter)
        
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)
        
        n = self.Pu.shape[1] 
        return np.r_[self.x[:-n],self.Pu@self.x[-n:]]






    



























