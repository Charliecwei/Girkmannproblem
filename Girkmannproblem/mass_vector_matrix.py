import numpy as np
from scipy.sparse import csr_matrix


def mass_vector_matrix(vspace,mu):
    E = 2.0*mu
    mesh = vspace.mesh
    boundary_cell_flag = mesh.ds.boundary_cell_flag()
    boundary_cell_index, = np.nonzero(boundary_cell_flag)
    inner_cell_index, = np.nonzero(~boundary_cell_flag)
    gdof = vspace.number_of_global_dofs()

    qf = vspace.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)
    
    ################################################
    ##############内部单元###############
    rho = mesh.bc_to_point(bcs,index=inner_cell_index)[...,0] #(NQ,NCin)
    phi = vspace.basis(bcs) #(NQ,1,ldof)

    #construct matrix
    rm = mesh.reference_cell_measure()
    D = mesh.first_fundamental_form(bcs,index=inner_cell_index)
    D = np.sqrt(np.linalg.det(D)) #(NQ,NCin)

    cell2dof = vspace.Incell2dof

    shape = cell2dof.shape+cell2dof.shape[1:] #(NCin,ldof,ldof)

    I = np.broadcast_to(cell2dof[:, :, None], shape=shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=shape)

    

    M = np.einsum('i,ij,ijk,ijl,ij->jkl',ws*rm,1.0/rho,phi,E*phi,D, optimize=True)
    In_M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))




    ##################################################
    ####################边界单元###################
    rho = mesh.bc_to_point(bcs,index=boundary_cell_index)[...,0] #(NQ,NCbd)
    phi = vspace.basis(bcs,is_bd_dof=True) #(NQ,1,ldof)

    #construct matrix
    rm = mesh.reference_cell_measure()
    D = mesh.first_fundamental_form(bcs,index=boundary_cell_index)
    D = np.sqrt(np.linalg.det(D)) #(NQ,NCbd)

    cell2dof = vspace.Bdcell2dof

    shape = cell2dof.shape+cell2dof.shape[1:] #(NCbd,ldof,ldof)
    
    I = np.broadcast_to(cell2dof[:, :, None], shape=shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=shape)


    M = np.einsum('i,ij,ijk,ijl,ij->jkl',ws*rm,1.0/rho,phi,E*phi,D, optimize=True)
    Bd_M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))

    M = In_M + Bd_M







    
    return M

