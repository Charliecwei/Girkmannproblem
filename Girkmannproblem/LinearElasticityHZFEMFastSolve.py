###############################胡张元程序快速求解算法############################
import numpy as np
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from timeit import default_timer as dtimer
from scipy.sparse import spdiags, tril, triu, csr_matrix, bmat
from fealpy.decorator import timer
from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace

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
            A = (M, B)
            F = (F0,F1) 

            离散的代数系统如下
            M x0 + B^T x1 = F0
            B x0          = F1

        '''
        self.M = A[0]
        self.B = A[1]
        self.C = A[2]
        self.F = np.r_[F[0],F[1].T.reshape(-1)]
        self.x = np.zeros_like(self.F)
        self.vspace = vspace
        self.mu = mu
        
        if lam == 0:
            self.D = self.M.diagonal()
        else:
            self.D = tspace.compliance_tensor_matrix(mu=mu,lam=0.0).diagonal()

        
        if isBDdof is not None:
            isfreedof, = np.nonzero(~isBDdof)
            isBDdof, = np.nonzero(isBDdof)
            self.x[isBDdof] = self.F[isBDdof]
           
            self.D = self.D[isfreedof]
            self.M = self.M[isfreedof][:,isfreedof]
            self.B = self.B[:,isfreedof]
            
            self.isfreedof = isfreedof
         
        else:
            self.isfreedof = np.arange(self.M.shape[0])

        self.isfreedof = np.r_[self.isfreedof,A[0].shape[0]+np.arange(A[1].shape[0])]
        self.F = self.F[self.isfreedof]

        tgdof = self.M.shape[0] 
        vgdof = self.B.shape[0]

        self.tgdof = tgdof
        self.vgdof = vgdof


        # S 相当于间断元的刚度矩阵
        S = self.B@spdiags(1/self.D,0,tgdof,tgdof)@self.B.T
        print(np.linalg.matrix_rank(S.todense())) 
        if self.C is not None:
            S += self.C

        self.smoother = GaussSeidelSmoother(S)
        self.S = S

        self.iter = 0

        
        #self.init_construct_amg_solver()
        
        #print(np.linalg.matrix_rank(self.S.todense()),self.S.shape)
        print(np.linalg.cond(self.S.todense()))
        ss
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
        S_coarse = cspace.stiff_matrix(isDDof=cspace.is_boundary_dof()) #粗空间S矩阵的逼近
        self.S_coarse = S_coarse[is_free_dof][:,is_free_dof]
        self.PI = self.PI[:,is_free_dof]


        self.PI1 = bmat([[self.PI,None],[None,self.PI]],format='csr')
        self.S_coarse1 = self.PI1.T@self.S@self.PI1

        
        S_coarse = cspace.linear_elasticity_matrix(0,self.mu)
        
        is_free_dof = np.r_[is_free_dof,is_free_dof+cspace.number_of_global_dofs()]
        self.S_coarse2 = S_coarse[is_free_dof][:,is_free_dof]
        





    def linear_operator(self,b):
        m = self.tgdof
        r = np.zeros_like(b)
        r[:m] = self.M@b[:m]+self.B.T@b[m:]
        r[m:] = self.B@b[:m]
        if self.C is not None:
            r[m:] -= self.C@r[m:]
        return r

    def precondieitoner(self,r):
        tgdof = self.tgdof

        u0 = r[:tgdof]/self.D
        #u0 = spsolve(self.M,r[:tgdof])
        r1 = r[tgdof:]-self.B@u0

        if True:
             u1 = spsolve(self.S,r1)

        elif True:

             u1 = np.zeros_like(r1)
             self.smoother.smooth(r1,u1,maxit=10)
            
             r2 = r1 - self.S@u1

             if False:
                 u1 += self.PI1@spsolve(self.S_coarse1,self.PI1.T@r2)

             elif True:
                 u1 += self.PI1@spsolve(self.S_coarse2,self.PI1.T@r2)

             elif True:
                 gdim = self.gdim
                 for i in range(gdim):
                     u1[i::gdim] += self.PI@(spsolve(self.S_coarse,self.PI.T@r2[i::gdim]))


             self.smoother.smooth(r1,u1,maxit=10)
             #self.smoother.smooth(r1,u1,lower=False,maxit=10)


        ####统计调用次数###
        self.iter += 1
        print(self.iter)
        ##################
        return np.r_[u0+self.B.T@u1/self.D, -u1]
    


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







