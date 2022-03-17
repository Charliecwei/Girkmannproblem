from PDE_Girkmann import Girkmann
from shell_ring import shell_ring
from shell_ring_s import shell_ring_s
from source_vector import source_vector
from set_essential_bc import set_essential_bc
from mass_vector_matrix import mass_vector_matrix
from compliance_tensor_matrix import compliance_tensor_matrix
from Post_processing import Post
from timeit import default_timer as dtimer #统计时间
from scipy.io import savemat


import numpy as np
from scipy.sparse import spdiags, bmat, csr_matrix, construct
from scipy.sparse.linalg import spsolve
import matlab.engine
import argparse
import matplotlib.pyplot as plt


from fealpy.functionspace.CurveHuZhangFiniteElementSpace2D_bubble import CurveHuZhangFiniteElementSpace_bubble
from fealpy.functionspace.DGLagrangeFiniteElementSpace_inner_bd import DGLagrangeFiniteElementSpace_inner_bd
from LinearElasticityHZFEMFastSolve_s import LinearElasticityHZFEMFastSolve, LinearElasticityHZFEMFastSolve_pure_stress #胡张元快速算法




##  参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上用胡张元求解Girkmannproblem
        """)


parser.add_argument('--degree',
        default=3, type=int,
        help='Lagrange 有限元空间的次数, 默认为 3 次.')



parser.add_argument('--mesh_h',
        default = 0.5, type=float,
        help='网格尺寸,默认为0.5')

parser.add_argument('--cormesh_h',
        default = 0.5, type=float,
        help='网格尺寸,默认为0.5')



parser.add_argument('--show_mesh',
        default = False, type=bool,
        help='是否展示网格，默认为False.')


parser.add_argument('--solver',
        default='direct',type=str,
        help='求解方程组')



args = parser.parse_args()
degree = args.degree
mesh_h = args.mesh_h
cormesh_h = args.cormesh_h
show_mesh = args.show_mesh 
solver = args.solver

##########Get PDE############
PDE = Girkmann()
#############################




#######general mesh#########

t0 = dtimer()
#mesh = shell_ring(cormesh_h,h=mesh_h,mdegree=3) 
mesh = shell_ring_s(cormesh_h,h=mesh_h,mdegree=3) 
print('gener mesh time:',dtimer()-t0)


if show_mesh:
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    #mesh.find_edge(axes,showindex=True)
    #mesh.find_cell(axes,showindex=True)
    #mesh.find_node(axes,showindex=True)
    #print(mesh.number_of_cells())
    #print(mesh.entity('cell')[-1])
    plt.show()
############################
#print(np.sum(~mesh.ds.boundary_node_flag()))
#s

print('construct matrix')
t0 = dtimer()
###################Load space###################
tspace = CurveHuZhangFiniteElementSpace_bubble(mesh, degree,degree+2)


vspace = DGLagrangeFiniteElementSpace_inner_bd(mesh, degree-1,degree+1)





tgdof = tspace.number_of_global_dofs()
vgdof = 2*vspace.number_of_global_dofs()

gdim = 2
sh = tspace.function()
uh = vspace.function(dim=gdim)
###################get matrix####################
M = compliance_tensor_matrix(tspace,PDE.mu)



B0,B1 = tspace.div_matrix(vspace)
C = mass_vector_matrix(vspace,PDE.mu)





###################get right hand################
F0 = np.zeros(tgdof,dtype=np.float64)
F1 =  source_vector(vspace,PDE)





##################stress boundary################
isBDdof = set_essential_bc(tspace,sh,PDE)
F0 -= M@sh
F0[isBDdof] = sh[isBDdof]
F1[:,0] -= B0@sh 
F1[:,1] -= B1@sh

bdIdx = np.zeros(tgdof, dtype=int)
bdIdx[isBDdof] = 1
Tbd = spdiags(bdIdx,0,tgdof,tgdof)
T = spdiags(1-bdIdx,0,tgdof,tgdof)
M = T@M@T + Tbd
B0 = B0@T
B1 = B1@T


print('construct matrix time:',dtimer()-t0)
##################slove##########################
if solver == 'direct':
    FF = np.r_[F0,F1.T.reshape(-1)]
    AA = bmat([[M, B0.transpose(), B1.transpose()],[B0, -C, None],[B1,None,None]],format='csr')



    print('solve matrix')
    t0 = dtimer()
    if len(FF)<100:
        x = spsolve(AA,FF)
    elif len(FF)<10000000:
        import transplant
        from fealpy.solver import MatlabSolver
        matlab = transplant.Matlab()
        solver = MatlabSolver(matlab)
        x = solver.divide(AA,FF)
    else:
        eng = matlab.engine.start_matlab()
        savemat('/Users/chen/Desktop/data.mat',{'AA':AA,'bb':FF})
        eng.addpath('/Users/chen/Desktop')
        x = np.array(eng.matlab_solve())[:,0]

elif solver == 'fast':
    B = construct.vstack([B0,B1],format='csr')
    C = bmat([[C,None],[None,csr_matrix(C.shape,dtype=C.dtype)]],format='csr')
    A = [M,B,C]
    F = [F0,F1]
    Fast_solver = LinearElasticityHZFEMFastSolve_pure_stress(A,F,vspace,tspace,PDE.mu,PDE.lam,isBDdof=isBDdof)
    
    x = Fast_solver.solve()
print('solve matrix time:',dtimer()-t0)












sh[:] = x[:tgdof]

print('Post')
t0 = dtimer()
Post = Post(PDE,sh,tspace)

print('gdof:',len(x))
print('Q:',Post.get_Q())
print('M:',Post.get_M())

print('Post time:',dtimer()-t0)

















