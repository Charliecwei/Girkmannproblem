import numpy as np



def set_essential_bc(tspace,sh,PDE):
    #最小二乘得到边界条件
    mesh = tspace.mesh
    index = mesh.ds.boundary_edge_index() #boundary index
    facebd2dof = tspace.Bdedge2dof #(NEbd,ldof,tdim)
    tdim = tspace.tensor_dimension()
    
    gdof = tspace.number_of_global_dofs()
    isBdDof = np.zeros(gdof,dtype=np.bool)
    
    pp = mesh.entity_barycenter('edge',index=index) #(NEbd,2)


    #idx, = np.nonzero(np.abs(pp[...,0])>1e-10)
    #isBdDof[facebd2dof[idx,:,1:]] = True

    #idx, = np.nonzero(np.abs(pp[...,0])<1e-10)
    #isBdDof[facebd2dof[idx,:,2]] = True

    isBdDof[facebd2dof[:,:,1:]] = True


    z = (PDE.r_0+PDE.d/2)*np.cos(PDE.alpha)-PDE.b
    idx, = np.nonzero(np.abs(pp[...,1]-z)<1e-10)

    
    bcs = tspace.space.bdbcs[idx] #(NEbd0,ldof)
    
 

    for i in range(len(idx)):
        pp = mesh.bc_to_point(bcs[i],index=[index[idx[i]]])[...,0,:] #(ldof,2)
        rho = pp[...,0] #(ldof,)
        sh[facebd2dof[idx[i],:,1]] = -PDE.p*rho
    

    ############################角点自由度特殊处理###################
    ipoint = tspace.dof.interpolation_points() #(NN,2)

    #############################找角点##################
    r_0 = PDE.r_0
    alpha = PDE.alpha
    d = PDE.d
    a = PDE.a
    b = PDE.b
    sin = np.sin
    cos = np.cos
    
    '''
    #(7,2)
    corner_point = np.array([[0,r_0-d/2],
                             [0,r_0+d/2],
                             [(r_0-d/2)*sin(alpha),(r_0-d/2)*cos(alpha)],
                             [(r_0+d/2)*sin(alpha),(r_0+d/2)*cos(alpha)],
                             [(r_0-d/2)*sin(alpha)+a, (r_0+d/2)*cos(alpha)],
                             [(r_0-d/2)*sin(alpha), (r_0+d/2)*cos(alpha)-b],
                             [(r_0-d/2)*sin(alpha)+a, (r_0+d/2)*cos(alpha)-b]],dtype=np.float64)
   
    idx = []
    for i in range(7):
        point = corner_point[i]

    '''
    idx = np.array([7,8,0,1,6,4,5])

    isBdDof[tdim*idx] = True #切向也是固定边界
    
    idx = idx[-2:] #(2,)
    Tensor_Frame = tspace.Tensor_Frame[idx[:,None]*tdim+np.arange(tdim)[None,:]] #(2,tdim,tdim)


    for i in range(2):
        if np.abs(Tensor_Frame[i,1,1]-1.0)>1e-10:
            sh[tdim*idx[i]+[0,1]] = sh[tdim*idx[i]+[1,0]]


    


    return isBdDof








