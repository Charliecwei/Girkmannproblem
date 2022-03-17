import gmsh
import sys
import numpy as np
from fealpy.mesh.CurveLagrangeTriangleMesh import CurveLagrangeTriangleMesh
from fealpy.pde.curve import curve_circle
import matplotlib.pyplot as plt





def circulmesh(h,mdegree=3):

    #############gmsh###########################
    gmsh.initialize()

    gmsh.model.geo.addPoint(0,0,0,h,1)
    gmsh.model.geo.addPoint(1,0,0,h,2)
    gmsh.model.geo.addPoint(0,1,0,h,3)
    gmsh.model.geo.addPoint(-1,0,0,h,4)
    gmsh.model.geo.addPoint(0,-1,0,h,5)



    #gmsh.model.geo.addLine(1,2,1)
    gmsh.model.geo.addCircleArc(2,1,3,1)
    gmsh.model.geo.addCircleArc(3,1,4,2)
    gmsh.model.geo.addCircleArc(4,1,5,3)
    gmsh.model.geo.addCircleArc(5,1,2,4)

    #gmsh.model.geo.addLine(3,1,3)

    #gmsh.model.geo.addCircleArc(3,1,4,3)

    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    gmsh.model.geo.addPlaneSurface([1], 1)







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


    mesh = CurveLagrangeTriangleMesh(node,cell,p=mdegree,curve=curve_circle())


    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes,showindex=True)
    mesh.find_edge(axes,showindex=True)
    mesh.find_cell(axes,showindex=True)
    plt.show()

    #################gmsh#################
    gmsh.finalize()

    return mesh
