import argparse
import numpy as np
import sympy as sp
import sys
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat, construct
from scipy.io import savemat
import scipy.io as sio

import matplotlib.pyplot as plt


#########Girkmann problem model########
from PDE_Girkmann import Girkmann




#####mesh model###########
from shell_ring import shell_ring
from ring import ring
from circul import circulmesh
###load mesh##############
from fealpy.mesh.CurveLagrangeTriangleMesh import CurveLagrangeTriangleMesh
###load function space
from fealpy.functionspace.CurveHuZhangFiniteElementSpace2D_bubble import CurveHuZhangFiniteElementSpace_bubble
from fealpy.functionspace.DGLagrangeFiniteElementSpace_inner_bd import DGLagrangeFiniteElementSpace_inner_bd







##  参数解析
parser = argparse.ArgumentParser(description=
        """
        检测Girkmannproblem网格剖分是否正确
        """)


parser.add_argument('--degree',
        default=3, type=int,
        help='Lagrange 有限元空间的次数, 默认为 3 次.')


parser.add_argument('--meshtype',
        default = 1,type=int, 
        help='区域类型')


parser.add_argument('--mesh_h',
        default = 0.5, type=float,
        help='网格尺寸,默认为0.5')




parser.add_argument('--show_mesh',
        default = False, type=bool,
        help='是否展示网格，默认为False.')


args = parser.parse_args()
sdegree = args.degree
meshtype = args.meshtype
mesh_h = args.mesh_h
show_mesh = args.show_mesh





###############lame coeffices#############

E = 2.059e10 #20.59Gpa

nu = 0

lam = E*nu/(1+nu)*(1-2*nu)
mu = E/(2*(1+nu))

##########################################


###############mesh#########################
if meshtype == 0:
    mesh = circulmesh(mesh_h,mdegree=3)
elif meshtype == 1:
    mesh = shell_ring(mesh_h,mdegree=3)
elif meshtype == 2:
    mesh = ring(mesh_h,mdegree=3)

############################################

###########################################
if show_mesh:
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_edge(axes,showindex=True)
    mesh.find_cell(axes,showindex=True)
    mesh.find_node(axes,showindex=True)
    plt.show()
###########################################








###load stress sapce and displacement space
tspace = CurveHuZhangFiniteElementSpace_bubble(mesh, sdegree,sdegree+2)
vspace = DGLagrangeFiniteElementSpace_inner_bd(mesh, sdegree-1,sdegree+1)


####construct matrix#####################
M = tspace.compliance_tensor_matrix(mu=mu,lam=lam)
B0,B1 = tspace.div_matrix(vspace)


AA = bmat([[M, B0.transpose(), B1.transpose()],[B0, None, None],[B1,None,None]],format='csr')


A = AA.todense()
print(np.linalg.cond(A))
print(A.shape,np.linalg.matrix_rank(A))
