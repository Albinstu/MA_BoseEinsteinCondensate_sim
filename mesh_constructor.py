#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:21:25 2026

@author: stu
"""

import numpy as np
from dolfinx import fem, mesh, plot, io
from mpi4py import MPI # for creating mesh
from dolfinx.io import gmsh as gmshio
import gmsh # mesh
import pyvista # plotting
import matplotlib.pyplot as plt
# from BEC_sim_besse_parallel import CAP_func, potential_func



# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
## DOMAIN PARAMETERS
box_width_comp_domain = 14 # of computational domain
box_height_comp_domain = 14 # -- || --

# rectangular domain is centered at origo
a = box_width_comp_domain * 0.5 # half axis of box width
b = box_height_comp_domain * 0.5 # half axis of box height
delta = 1.5 # change to larger maybe 1 or 2


# potential coordinates; row 0 - potential x-coord bounds
#                        row 1 - potential y-coord bounds
potential_coords = [(0.7 + box_width_comp_domain * 0.25 - a, 
                    box_width_comp_domain * 0.25 - a \
                    # + 2.5/8 * box_width_comp_domain), 
                    + 2), 
                   (-b, b)]

# computational_domain_coord_ranges = [(-a, a),
#                                      (-b, b)]

# coordinate ranges extended domain : upper row x-coords, lower row y-coords
ext_domain_coord_ranges = [(-a-delta, a + delta),
                                (-b - delta, b + delta)]

# coordinates of polygon vertices for extended domain
ext_domain_polygon_vertices = [(-a - delta, -b - delta),
                                    (-a - delta, b + delta),
                                    (a + delta, b + delta),
                                    (a + delta, -b - delta)]

# init mesh size
h = 0.0




# =============================================================================
# FUNCTIONS
# =============================================================================
def polygon_mesh(vertices, h):
    gmsh.initialize()
    gmsh.model.add("polygon")

    # Points
    pts = [gmsh.model.geo.addPoint(x, y, 0.0) for x, y in vertices]

    # Lines
    lines = [
        gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
        for i in range(len(pts))
    ]

    # Surface
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # Physical groups (required)
    gmsh.model.addPhysicalGroup(2, [surface], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")

    gmsh.model.addPhysicalGroup(1, lines, 2)
    gmsh.model.setPhysicalName(1, 2, "boundary")

    gmsh.model.mesh.generate(2)

    # NEW API: returns MeshData object
    meshdata = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0)

    gmsh.finalize()

    return meshdata.mesh, meshdata.cell_tags, meshdata.facet_tags


def domain_param_writer():
    np.savez("sim_data_files/box_params.npz",
             bwcd=box_width_comp_domain,
             bhcp=box_height_comp_domain,
             a_axis=a,
             b_axis=b,
             d=delta,
             pot_coords=potential_coords,
             domain_coord_ranges=ext_domain_coord_ranges,
             domain_polygon_vertices=ext_domain_polygon_vertices,
             h_max=h
             )
    return;


def mesh_constructor_n_writer(h):
    ext_domain_mesh, _, _ = polygon_mesh(ext_domain_polygon_vertices, h)

    with io.XDMFFile(ext_domain_mesh.comm, "sim_data_files/mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(ext_domain_mesh)

    xdmf.close()
    
    return;



def main():
    h = 0.02 # max mesh size
    mesh_constructor_n_writer(h)
    
    domain_param_writer()
    
    return 0



if __name__ == "__main__":
    main()
    










# =============================================================================
# TEST FUNCTIONS WOKRING CORRECTLY WITH DOLFINX 
# 
# V = fem.functionspace(ext_domain_mesh, ("Lagrange", 1))
# Q = fem.functionspace(ext_domain_mesh, ("DG", 0))
# 
# CAP_W = fem.Function(Q)
# CAP_W.interpolate(CAP_func)
# 
# pot_V = fem.Function(Q)
# pot_V.interpolate(potential_func)
# 
# plot_mesh(ext_domain_mesh, CAP_W.x.array)
# plot_mesh(ext_domain_mesh, pot_V.x.array)
# =============================================================================

