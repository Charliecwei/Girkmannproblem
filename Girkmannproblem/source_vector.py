import numpy as np




def source_vector(vspace,PDE):
     mesh = vspace.mesh
     gdof = vspace.number_of_global_dofs()
     b = np.zeros((gdof,2),dtype=np.float64)

     boundary_cell_flag = mesh.ds.boundary_cell_flag()
     boundary_cell_index, = np.nonzero(boundary_cell_flag)
     inner_cell_index, = np.nonzero(~boundary_cell_flag)
     
     bcs, ws = vspace.integrator.get_quadrature_points_and_weights()
     ####################################################
     ##################内部单元#####################
     pp = mesh.entity_barycenter('cell',index=inner_cell_index)
     alpha = PDE.get_alpha(pp)
     idx, = np.nonzero((alpha < PDE.alpha)&(alpha>0.0))#(NCin,ldof)
     
     rho = mesh.bc_to_point(bcs,index=inner_cell_index[idx])[...,0] #(NQ,NCin)
     phi = vspace.basis(bcs)#(NQ,1,ldof)

     #construct vector
     rm = mesh.reference_cell_measure()
     D = mesh.first_fundamental_form(bcs,index=inner_cell_index[idx])
     D = np.sqrt(np.linalg.det(D)) #(NQ,NCin)

     bb = np.einsum('m,mi,mik,mi->ik',ws*rm,rho,phi,D)*PDE.F #(NCin,ldof)
     cell2dof = vspace.Incell2dof[idx] #(NCin,ldof)
    
     np.add.at(b[:,1],cell2dof,bb)


     ####################################################
     #################边界单元#######################
     pp = mesh.entity_barycenter('cell',index=boundary_cell_index)
     alpha = PDE.get_alpha(pp)
     idx, = np.nonzero((alpha < PDE.alpha)&(alpha>0.0))#(NCbd,ldof)

     rho = mesh.bc_to_point(bcs,index=boundary_cell_index[idx])[...,0] #(NQ,NCbd)
     phi = vspace.basis(bcs,is_bd_dof=True)#(NQ,1,ldof)
     

     #construct vector
     rm = mesh.reference_cell_measure()
     D = mesh.first_fundamental_form(bcs,index=boundary_cell_index[idx])
     D = np.sqrt(np.linalg.det(D)) #(NQ,NCbd)

     bb = np.einsum('m,mi,mik,mi->ik',ws*rm,rho,phi,D)*PDE.F #(NCbd,ldof)
     cell2dof = vspace.Bdcell2dof[idx] #(NCbd,ldof)

     np.add.at(b[:,1],cell2dof,bb)





     return b
