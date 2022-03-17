import gmsh
import sys
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.mesh.CurveLagrangeTriangleMesh import CurveLagrangeTriangleMesh
from Curve_boundary import Girk_curve
import matplotlib.pyplot as plt



def shell_ring_s(hs, h=1.0, mdegree=3):
    
    

    pi = np.pi
    sin = np.sin
    cos = np.cos

    rho_0 = 15.0
    alpha = 2*pi/9
    r_0 = rho_0/sin(alpha)
    d = 0.06
    a = 0.60
    b = 0.50

    alphas = pi/180 #辅助用，保证边界单元只有一条边界边

    curve = Girk_curve()






    ################gmesh################

    ########shell model#########
    gmsh.initialize()

    ###########piont#################
    gmsh.model.geo.addPoint(0,0,0,h,1)
    
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha),(r_0-d/2)*cos(alpha),0,hs,2)
    gmsh.model.geo.addPoint((r_0+d/2)*sin(alpha),(r_0+d/2)*cos(alpha),0,hs,3)
    
    gmsh.model.geo.addPoint((r_0+d/2)*sin(alphas),(r_0+d/2)*cos(alphas),0,h,4)
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alphas),(r_0-d/2)*cos(alphas),0,h,5)




    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha), (r_0+d/2)*cos(alpha)-b, 0, h, 6)
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha)+a, (r_0+d/2)*cos(alpha)-b, 0, h, 7)
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha)+a, (r_0+d/2)*cos(alpha), 0, h, 8)






    gmsh.model.geo.addPoint(0,(r_0-d/2),0,h,9)
    gmsh.model.geo.addPoint(0,(r_0+d/2),0,h,10)
    gmsh.model.geo.addPoint(0.5*r_0*sin(alphas),0.5*r_0*(1+cos(alphas)),0,h,11)



    
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha-alphas/10),(r_0-d/2)*cos(alpha-alphas/10),0,h,12)
    gmsh.model.geo.addPoint((r_0+d/2)*sin(alpha-alphas/10),(r_0+d/2)*cos(alpha-alphas/10),0,h,13)

    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha), (r_0+d/2)*cos(alpha)-d*cos(alpha)-b/10, 0, h, 14)
    gmsh.model.geo.addPoint((r_0-d/2)*sin(alpha)+d*sin(alpha)+a/10,(r_0+d/2)*cos(alpha), 0, h, 15)


    ###################point######################


    ##################line##########################
    gmsh.model.geo.addCircleArc(12,1,2,1)
    gmsh.model.geo.addLine(2,3,2)
    gmsh.model.geo.addCircleArc(3,1,13,3)
    gmsh.model.geo.addLine(13,12,4)    

    gmsh.model.geo.addLine(2,14,5)
    gmsh.model.geo.addLine(14,15,6)
    gmsh.model.geo.addLine(15,3,7)


    gmsh.model.geo.addLine(14,6,8)
    gmsh.model.geo.addLine(6,7,9)
    gmsh.model.geo.addLine(7,8,10)
    gmsh.model.geo.addLine(8,15,11)


    gmsh.model.geo.addCircleArc(13,1,4,12)
    gmsh.model.geo.addLine(4,5,13)
    gmsh.model.geo.addCircleArc(5,1,12,14)




    gmsh.model.geo.addLine(9,5,15)
    gmsh.model.geo.addLine(4,10,16)
    gmsh.model.geo.addLine(10,9,17)

    gmsh.model.geo.addLine(9,11,18)
    gmsh.model.geo.addLine(5,11,19)
    gmsh.model.geo.addLine(4,11,20)
    gmsh.model.geo.addLine(10,11,21)    
    ###################line########################



    #################surface######################
    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    gmsh.model.geo.addPlaneSurface([1],1)

    
    gmsh.model.geo.addCurveLoop([5,6,7,-2],2)
    gmsh.model.geo.addPlaneSurface([2], 2)
    

    gmsh.model.geo.addCurveLoop([8,9,10,11,-6],3)
    gmsh.model.geo.addPlaneSurface([3],3)



    gmsh.model.geo.addCurveLoop([12,13,14,-4],4)
    gmsh.model.geo.addPlaneSurface([4],4)
    

    gmsh.model.geo.addCurveLoop([15,19,-18],5)
    gmsh.model.geo.addPlaneSurface([5],5)

    gmsh.model.geo.addCurveLoop([-13,20,-19],6)
    gmsh.model.geo.addPlaneSurface([6],6)

    gmsh.model.geo.addCurveLoop([16,21,-20],7)
    gmsh.model.geo.addPlaneSurface([7],7)

    gmsh.model.geo.addCurveLoop([17,18,-21],8)
    gmsh.model.geo.addPlaneSurface([8],8)


    #################surface######################






















    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    #gmsh.fltk.run()


    node = gmsh.model.mesh.getNodes()[1]
    node = node[3:] #第一个点（0，0，0）不是网格点，不要
    #print(node[:3])
    NN = node.shape[0]//3
    node = node.reshape(NN,3)
    node = np.array(node[:,:2],dtype=np.float64)

    cell = np.array(gmsh.model.mesh.getElements(dim=2)[2][0]-1-1,dtype=np.int64)
    NC = cell.shape[0]//3
    cell = cell.reshape(NC,3)

    #print(np.min(cell))



    #mesh = TriangleMesh(node,cell)
    mesh = CurveLagrangeTriangleMesh(node,cell,p=mdegree,curve=curve)


    if (np.min(mesh.entity_measure('cell'))<0):
        raise ValueError("Cell is not right!")

    #fig = plt.figure()
    #axes = fig.gca()
    #mesh.add_plot(axes)
    #mesh.find_node(axes,showindex=True)
    #mesh.find_edge(axes,showindex=True)
    #mesh.find_cell(axes,showindex=True)
    #plt.show()




    ################gmesh################
    gmsh.finalize()

    return mesh
