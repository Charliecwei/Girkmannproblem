import numpy as np
from scipy.sparse import csr_matrix
import pdb




def compliance_tensor_matrix(tspace,mu):
    E = 2.0*mu
    mesh = tspace.mesh
    tdim = tspace.tensor_dimension()
    gdim = tspace.geo_dimension()
    boundary_cell_flag = mesh.ds.boundary_cell_flag()
    
    ####################################################
    ######内部单元#########
    inner_cell_index, = np.nonzero(~boundary_cell_flag)
    ldof = tspace.number_of_local_dofs()
    bcs, ws = tspace.integrator.quadpts, tspace.integrator.weights
    rho = tspace.mesh.bc_to_point(bcs,index=inner_cell_index)[...,0] #(NQ,NCin)

    NCin = tspace.Incell2dof.shape[0]
    NQ = bcs.shape[0]
    phi = tspace.basis(bcs).reshape(NQ,NCin,-1,tdim)#(NQ,NC,ldof,tdim)
    

    #construct matrix
    d = np.array([1, 1, 2])
    rm = mesh.reference_cell_measure()
    D = mesh.first_fundamental_form(bcs,index=inner_cell_index)
    D = np.sqrt(np.linalg.det(D)) #(NQ,NCin)    

    M = np.einsum('i, ij, ijkm, m, ijom, ij->jko', ws*rm, 1.0/rho, phi/E, d, phi, D, optimize=True) #(NCin,ldof,ldof)
    #Ms = np.einsum('i,  ijkm, m, ijom, ij->jko', ws*rm,  phi/E, d, phi, D, optimize=True) #(NCin,ldof,ldof)


    I = np.einsum('ij, k->ijk', tspace.Incell2dof.reshape(NCin,-1), np.ones(ldof))
    J = I.swapaxes(-1, -2)
    tgdof = tspace.number_of_global_dofs()
    In_M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(tgdof, tgdof))
    #In_Ms = csr_matrix((Ms.flat, (I.flat, J.flat)), shape=(tgdof, tgdof)) 

    #####################################################
    ############边界单元#############
    boundary_cell_index, =  np.nonzero(boundary_cell_flag)
    ldof = tspace.number_of_local_dofs(is_bd_dof=True)
    qf = mesh.integrator(11,'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    #bcs, ws = tspace.integrator.quadpts, tspace.integrator.weights
    rho = tspace.mesh.bc_to_point(bcs,index=boundary_cell_index)[...,0] #(NQ,NCbd)

    NCbd = tspace.Bdcell2dof.shape[0]
    NQ = bcs.shape[0]
    phi = tspace.basis(bcs,is_bd_dof=True).reshape(NQ,NCbd,-1,tdim) #(NQ,NCbd,ldof,tdim)

    d = np.array([1, 1, 2])
    rm = mesh.reference_cell_measure()
    D = mesh.first_fundamental_form(bcs,index=boundary_cell_index)
    D = np.sqrt(np.linalg.det(D))
    
    M = np.einsum('i, ij, ijkm, m, ijom, ij->jko', ws*rm, 1.0/rho, phi/E, d, phi, D, optimize=True) #(NCbd,ldof,ldof)
    #Ms = np.einsum('i,  ijkm, m, ijom, ij->jko', ws*rm,  phi/E, d, phi, D, optimize=True) #(NCbd,ldof,ldof)
    

    I = np.einsum('ij, k->ijk', tspace.Bdcell2dof.reshape(NCbd,-1), np.ones(ldof))
    J = I.swapaxes(-1, -2)
    tgdof = tspace.number_of_global_dofs()
    Bd_M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(tgdof, tgdof))
    #Bd_Ms = csr_matrix((Ms.flat, (I.flat, J.flat)), shape=(tgdof, tgdof))



    M = In_M + Bd_M
    #Ms = In_Ms + Bd_Ms

    
    #print(np.max(np.abs(phi)),np.min(np.abs(phi)))

    #print(np.max(M.diagonal()),np.min(M.diagonal()))
    #print(np.max(Ms.diagonal()),np.min(Ms.diagonal()))
    #print(np.max(1/rho)/np.min(1/rho))
    
    #pdb.set_trace()
    #sss
    return M
