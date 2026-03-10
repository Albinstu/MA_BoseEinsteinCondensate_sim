#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:39:22 2026

@author: stu
"""

import numpy as np
from dolfinx import fem, mesh, plot, default_scalar_type
from mpi4py import MPI # for creating mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem #, NonlinearProblem
from dolfinx.io import gmsh as gmshio
import gmsh # mesh
import ufl
import pyvista # plotting
import matplotlib.pyplot as plt






# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == "c"

# CHECK IF ALGEBRA PETSc CAN HANDLE COMPLEX
if np.issubdtype(PETSc.ScalarType, np.complexfloating):
    # imaginaryUnit = PETSc.ScalarType(0 + 1j)
    j_im = PETSc.ScalarType(0 + 1j)
else:
    print('Complex Treatment is needed')
    exit() # not correct way to handle

## DOMAIN PARAMETERS
# box_width_comp_domain = 16 # of computational domain
# box_height_comp_domain = 9 # -- || --
box_width_comp_domain = 16 # of computational domain
box_height_comp_domain = 10 # -- || --

# rectangular domain is centered at origo
a = box_width_comp_domain * 0.5 # half axis of box width
b = box_height_comp_domain * 0.5 # half axis of box height
delta = 1 # change to larger maybe 1 or 2

# potential coordinates; row 0 - potential x-coord bounds
#                        row 1 - potential y-coord bounds
potential_coords = [(box_width_comp_domain * 0.25 - a, 
                    box_width_comp_domain * 0.25 - a 
                    + 2.5/8 * box_width_comp_domain), 
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


## EQUATION COEFFICIENT PARAMETERS
u = 2 # velocity of coupled term
g = 0.8 # scattering amplitude 
g1 = g # IF CHANGE VALUE HERE, MUST CHAGNE VALUE IN NON-LINEAR FUNC
g2 = g
g3 = g
# b1 = np.array([1, -1j]) # convection vectors
# b2 = np.array([1, 1j]) # ---- || ----

T = 1 # end time
N = 500 # number of time steps from 0th
dt = T / N # time step
dt_inv = N / T


# radius of start circ
start_circ_r = np.abs(-a + potential_coords[0][0]) * 0.5
ang_of_atac = 0.0 # angle of attack against normal to potential barriers centre

# initial coordinates of wave packet
initial_coord = np.array([- start_circ_r * np.cos(ang_of_atac), 
                 start_circ_r * np.sin(ang_of_atac)])

# momentum vector of initial cond
# p_vec = - 20 * initial_coord[:] / np.linalg.norm(initial_coord)
p_vec = - 5 * (initial_coord[:] / np.linalg.norm(initial_coord))
# p_vec = [1, 0] # velocity of initial cond

gauss_width = 1 # arbitrarily chosen # smaller diffuses faster
state_pop1 = 1 / np.sqrt(2) # state population
# state_pop1 = 1 # state population
state_pop2 = state_pop1



# =============================================================================
# FUNCTIONS
# =============================================================================
def initial_condition(x, state_pop, init_coord=initial_coord, p_vec=p_vec):
    consts = state_pop / ( gauss_width * np.sqrt(np.pi))
    gauss_pack = np.exp(- ( (x[0] - init_coord[0])**2 \
       + (x[1] - init_coord[1])**2 ) / (2 * gauss_width ** 2) )
    
    wave_prop = np.exp( 1j * ( p_vec[0] * (x[0] - init_coord[0]) \
                            + p_vec[1] * (x[1] - init_coord[1]) ))
    
    return consts * gauss_pack * wave_prop


def plot_mesh(mesh: mesh.Mesh, values=None):
    """
    Given a DOLFINx mesh, create a pyvista.UnstructuredGrid,
    and plot it and the mesh nodes.

    Args:
        mesh: The mesh we want to visualize
        values: List of values indicating a marker for each cell in the mesh

    Note:
        If values are given as input, they are assumed to be a marker
        for each cell in the domain.
    """
    # We create a pyvista plotter instance
    # pyvista.OFF_SCREEN=True #
    # pyvista.set_jupyter_backend('static')
    plotter = pyvista.Plotter()

    # Since the meshes might be created with higher order elements,
    # we start by creating a linearized mesh for nicely inspecting the triangulation.
    V_linear = fem.functionspace(mesh, ("Lagrange", 1))
    linear_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_linear))

    # If the mesh is higher order, we plot the nodes on the exterior boundaries,
    # as well as the mesh itself (with filled in cell markers)
    if mesh.geometry.cmap.degree > 1:
        ugrid = pyvista.UnstructuredGrid(*plot.vtk_mesh(mesh))
        if values is not None:
            ugrid.cell_data["Marker"] = values
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        plotter.add_mesh(ugrid, show_edges=False)
        plotter.add_mesh(linear_grid, style="wireframe", color="black")
    else:
        # If the mesh is linear we add in the cell markers
        if values is not None:
            linear_grid.cell_data["Marker"] = values
        plotter.add_mesh(linear_grid, show_edges=True)

    # We plot the coordinate axis and align it with the xy-plane
    plotter.show_axes()
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        img = plotter.screenshot(return_img=True)

        # Show it inline
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def polygon_mesh(vertices, h):
    # MAKE THIS BE ABLE TO RUN IN PARALLEL
    # MAKE THIS BE ABLE TO RUN IN PARALLEL
    # MAKE THIS BE ABLE TO RUN IN PARALLEL
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


def CAP_func(x): # Complex absorbing potential
    abs_x = np.abs(x[0]) # - a to optimise
    abs_y = np.abs(x[1]) # - b to optimise
    n = 2
    CAP_strength = 50
    index_func_x = (abs_x > a) # gep ?
    index_func_y = (abs_y > b) # geq ?
    inv_delta = 1 / delta
    
    CAP_x = index_func_x * ((abs_x - a) * inv_delta) ** n
    CAP_y = index_func_y * ((abs_y - b) * inv_delta) ** n
    
    return CAP_strength * (CAP_x + CAP_y)
    

def potential_func(x): # potential function
    pot_strength = 5 # potential strength
    
    # extraction of potential region bounds
    pot_x_range = potential_coords[0]
    pot_y_range = potential_coords[1]
    
    # construct in potential region bool
    bool_in_pot_region = ( x[0] >= pot_x_range[0] ) *\
        ( x[0] <= pot_x_range[1] ) *\
            ( x[1] >= pot_y_range[0] ) *\
            ( x[1] <= pot_y_range[1])

    return pot_strength * bool_in_pot_region


def F_nonlin_func(psi1_arr, psi2_arr):
    # MODIFY THIS FUNC IF g1,g2,g3 CHANGE VALUES
    p1 = np.conj(psi1_arr) * g * ( psi1_arr + psi2_arr ) 
    p2 = np.conj(psi2_arr) * g * ( psi1_arr + psi2_arr )
    
    return p1 + p2


def problem_solver(domain_mesh):
    pass
    


# =============================================================================
# SOLVER CODE
# =============================================================================
ext_domain_mesh, _, _ = polygon_mesh(ext_domain_polygon_vertices, 0.1)
# plot_mesh(ext_domain_mesh)



# DISCRETE SPACES
# function space
V = fem.functionspace(ext_domain_mesh, ("Lagrange", 1)) 
# coeff func space; one-to-one connection cell and value
Q = fem.functionspace(ext_domain_mesh, ("DG", 0)) 


# DEFINE COEFF
CAP_W = fem.Function(Q)
CAP_W.name = "CAP = W"
CAP_W.interpolate(CAP_func)

pot_V = fem.Function(Q)
pot_V.name = "V"
pot_V.interpolate(potential_func)


# APPLY BC
tdim = ext_domain_mesh.topology.dim # give dim of cells
fdim = tdim - 1 # dim of facets/edges
ext_domain_mesh.topology.create_connectivity(fdim, tdim) # edge to cell
boundary_facets = mesh.exterior_facet_indices(ext_domain_mesh.topology) 
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bcs = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)


# CONSTRUCT FUNCTIONS
psi1_old = fem.Function(V)
psi2_old = fem.Function(V)

psi1_new = fem.Function(V)
psi2_new = fem.Function(V)

# steppers part of relaxation scheme
gamma_new = fem.Function(V)
phi1_new = fem.Function(V)
phi2_new = fem.Function(V)


# trial & test function
PHI = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
v_conj = ufl.conj(v) # conjugate needed instead of ufl.inner


b1_vec = ufl.as_tensor([1.0 + 0.0j, -1.0j, 0.0j]) # WORKS AND IS CLEAN
b2_vec = ufl.as_tensor([1.0 + 0.0j, 1.0j, 0.0j]) # WORKS AND IS CLEAN




# create non_linear func
# IF VALUE OF g1,g2,g3 IS CHANGED THEN MUST REWRITE FUNCTIONS
# F_nonlin_func = lambda psi1, psi2 : ufl.conj(psi1) * g * ( psi1 + psi2 ) +\
#     ufl.conj(psi2) * g * ( psi1 + psi2 )


# INIT
psi1_old.interpolate(lambda x : initial_condition(x, state_pop1))
psi2_old.interpolate(lambda x : initial_condition(x, state_pop2))


gamma_new.x.array[:] = F_nonlin_func(psi1_old.x.array[:], psi2_old.x.array[:])
phi1_new.x.array[:] = psi2_old.x.array[:]
phi2_new.x.array[:] = psi1_old.x.array[:]



# CONSTRUCT BILINEAR AND LINEAR FORMS
a1_const = (j_im * 2 * dt_inv - pot_V + j_im * CAP_W) * PHI * v_conj * ufl.dx
a1_const += - 0.5 * ufl.inner( ufl.grad(PHI), ufl.grad(v)) * ufl.dx
a1 = - gamma_new * PHI * v_conj * ufl.dx + a1_const

L1 = j_im * 4 * dt_inv * psi1_old * v_conj * ufl.dx
L1 += - j_im * u * ufl.dot(b1_vec, ufl.grad(phi1_new + psi2_old)) * v_conj * ufl.dx

prob1 = LinearProblem(
    a1,
    L1,
    bcs=[bcs],
    u=psi1_new,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, 
    # petsc_options = { # best for medium 
    # "ksp_type": "gmres",
    # "pc_type": "ilu"
    #             },
    # petsc_options = { # best for large parallel by chatgpt "FEniCSx complex conjugate..."
    # "ksp_type": "gmres",
    # "pc_type": "hypre",
    # "pc_hypre_type": "boomeramg"
    # },
    petsc_options_prefix="psi1_solver_",
    )




a2_const = (j_im * 2 * dt_inv - pot_V + j_im * CAP_W) * PHI * v_conj * ufl.dx
a2_const += - 0.5 * ufl.inner( ufl.grad(PHI), ufl.grad(v)) * ufl.dx
a2 = - gamma_new * PHI * v_conj * ufl.dx + a2_const

L2 = j_im * 4 * dt_inv * psi2_old * v_conj * ufl.dx
L2 += - j_im * u * ufl.dot(b2_vec, ufl.grad(phi2_new + psi1_old)) * v_conj * ufl.dx

prob2 = LinearProblem(
    a2,
    L2,
    bcs=[bcs],
    u=psi2_new,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, 
    # petsc_options = { # best for medium 
    # "ksp_type": "gmres",
    # "pc_type": "ilu"
    #             },
    # petsc_options = { # best for large parallel by chatgpt "FEniCSx complex conjugate..."
    # "ksp_type": "gmres",
    # "pc_type": "hypre",
    # "pc_hypre_type": "boomeramg"
    # },
    petsc_options_prefix="psi1_solver_",
    )


########### PLOTTER SETUP ############
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

grid.point_data["Psi"] = np.abs(psi1_new.x.array)**2 \
    + np.abs(psi1_new.x.array)**2
# warped = grid.warp_by_scalar("Psi", factor=1)

viridis = plt.colormaps.get_cmap("viridis").resampled(1000)
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)

renderer = plotter.add_mesh(
    # warped,
    grid,
    # show_edges=True,
    show_edges=False,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[0, max(np.abs(psi1_new.x.array)**2 + np.abs(psi2_new.x.array)**2)],
)
plotter.view_xy()
# plotter.camera.tight()
plotter.camera.zoom(1.2)


for _ in range(N//2):
    
    # solve problem
    prob1.solve()
    prob2.solve()
    
    # update solutions
    psi1_new.x.array[:] = psi1_new.x.array[:] - psi1_old.x.array[:]
    psi1_old.x.array[:] = psi1_new.x.array[:]
    
    psi2_new.x.array[:] = psi2_new.x.array[:] - psi2_old.x.array[:]
    psi2_old.x.array[:] = psi2_new.x.array[:]
    
    
    gamma_new.x.array[:] = - gamma_new.x.array
    gamma_new.x.array[:] += 2*F_nonlin_func(psi1_old.x.array[:], psi2_old.x.array[:])
    
    phi1_new.x.array[:] = 2 * psi2_old.x.array[:] - phi1_new.x.array[:]
    phi2_new.x.array[:] = 2 * psi1_old.x.array[:] - phi2_new.x.array[:]

    prob = np.abs(psi1_new.x.array)**2 + np.abs(psi2_new.x.array)**2
    
    # new_warped = grid.warp_by_scalar("Psi", factor=1)
    # warped.points[:, :] = new_warped.points
    # warped.point_data["Psi"][:] = prob
    
    grid.point_data["Psi"][:] = prob
    
    plotter.write_frame()

plotter.close()


# L2 norm
comm = psi1_new.function_space.mesh.comm
PSI_norm = fem.form((psi1_new * ufl.conj(psi1_new) \
                     + psi2_new * ufl.conj(psi2_new))* ufl.dx)
PSI_norm_glob = np.sqrt(comm.allreduce(fem.assemble_scalar(PSI_norm), MPI.SUM))
print(PSI_norm_glob)


# =============================================================================
# FURTHER GOAL: 
#   PARALLELISE THE SOLVER
# =============================================================================



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


# =============================================================================
# TEST OF FUNCTIONS; WORKING
# 
# X_potential_in = [ potential_coords[0][0] + delta, potential_coords[1][1] ]
# X_potential_out = [ X_potential_in[0], X_potential_in[1] + delta ]
# 
# print(CAP(X_potential_in) == 0) # true
# print(CAP(X_potential_out) > 0) # true
# 
# print()
# 
# print(potential_func(X_potential_in) > 0) # true
# print(potential_func(X_potential_out) == 0) # true
# =============================================================================