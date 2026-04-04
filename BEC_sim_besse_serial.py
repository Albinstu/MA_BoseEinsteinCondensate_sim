#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:39:22 2026

@author: stu
"""

import numpy as np
from dolfinx import fem, mesh, plot, default_scalar_type, io
from dolfinx.fem import petsc
from dolfinx.io import gmsh as gmshio
from dolfinx.la.petsc import create_vector_wrap
from mpi4py import MPI # for creating mesh
from petsc4py import PETSc
import gmsh # mesh
import ufl
import pyvista # plotting
import matplotlib.pyplot as plt
from time import perf_counter # preformance measure




# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

## DOMAIN PARAMETERS
box_width_comp_domain = None
box_height_comp_domain = None
a = None
b = None
delta = None
potential_coords = None
ext_domain_coord_ranges = None
ext_domain_polygon_vertices = None
h = None

u = 0 # velocity of coupled term
g = 0.0 # scattering amplitude 
g1 = g # IF CHANGE VALUE HERE, MUST CHAGNE VALUE IN NON-LINEAR FUNC
g2 = g
g3 = g
# radius of start circ
start_circ_r = 0.0
ang_of_atac = 0.0 # angle of attack against normal to potential barriers centre
# initial coordinates of wave packet
initial_coord = np.array([- start_circ_r * np.cos(ang_of_atac), 
                 start_circ_r * np.sin(ang_of_atac)])
p_vec = np.copy(initial_coord)
gauss_width = 0.8 # arbitrarily chosen # smaller diffuses faster
state_pop1 = 1 / np.sqrt(2) # state population
state_pop2 = - state_pop1




# =============================================================================
# FUNCTIONS
# =============================================================================
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


def plotter_func(domain_mesh, sim_solutions,
                 sim_time_iter_var, scale_factor=10, args=None):
    
    # UNPACK variables
    # T = sim_time_iter_var[0]
    N_iter = sim_time_iter_var[1]
    # dt = sim_time_iter_var[2]
    # dt_inv = sim_time_iter_var[3]
    
    
    
    # if not same length then something wrong
    assert (N_iter + 1) == (sim_solutions.shape[0] // 2) #debug
    
    
    V = fem.functionspace(domain_mesh, ("Lagrange", 1))
    
        
    psi1_t = sim_solutions[0]
    psi2_t = sim_solutions[N_iter+1]
    
    
    ########### PLOTTER SETUP ############
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
    pyvista.OFF_SCREEN = True
    
    plotter = pyvista.Plotter()
    plotter.window_size = [2048, 1536] # doubles window size
    plotter.open_gif("PSI_evolution.gif", fps=N // 10)
    
    grid.point_data["Psi"] = np.abs(psi1_t)**2 \
        + np.abs(psi2_t)**2
    # warped = grid.warp_by_scalar("Psi", factor=1)
    
    viridis = plt.colormaps.get_cmap("viridis").resampled(1000)
    sargs = dict(
        title_font_size=25,
        label_font_size=20,
        fmt="%.2e",
        color="black",
        position_x=0.9,
        position_y=0.1,
        width=0.1,
        height=0.8,
        vertical=True
    )
    
    renderer = plotter.add_mesh(
        # warped,
        grid,
        # show_edges=True,
        show_edges=False,
        lighting=False,
        cmap=viridis,
        scalar_bar_args=sargs,
        clim=[0, 
              scale_factor * max(np.abs(psi1_t)**2 + np.abs(psi2_t)**2)
              ],
    )
    
    
    # SHADY BUT MAY WORK
    # POT. FIX, DRAW MANUAL RECTANGLES
    # SAVE DOMAIN DATA IN np FILE WHICH IS READ BEFORE PLOTTING
    if args != None :
        x0, x1 = potential_coords[0]
        y0, y1 = potential_coords[1]
        
        
        pot_edges = [
            pyvista.Line((x0,y0,0),(x1,y0,0)),
            pyvista.Line((x1,y0,0),(x1,y1,0)),
            pyvista.Line((x1,y1,0),(x0,y1,0)),
            pyvista.Line((x0,y1,0),(x0,y0,0))
            ]
        
        
        xmin, xmax = ext_domain_coord_ranges[0]
        ymin, ymax = ext_domain_coord_ranges[1]
        
        xmin += delta
        xmax += - delta
        ymin += delta
        ymax += - delta
        
        CAP_edges = [
            pyvista.Line((xmin,ymin,0),(xmax,ymin,0)),
            pyvista.Line((xmax,ymin,0),(xmax,ymax,0)),
            pyvista.Line((xmax,ymax,0),(xmin,ymax,0)),
            pyvista.Line((xmin,ymax,0),(xmin,ymin,0))
        ]
        
        
        for i in range(4):
            plotter.add_mesh(CAP_edges[i], 
                             color="white", 
                             line_width=2,
                             opacity=0.7
                             )
            
            plotter.add_mesh(pot_edges[i], 
                             color="red", 
                             line_width=2,
                             opacity=0.7
                             )
    
    
    plotter.view_xy()
    plotter.camera.zoom(1.38)
        
    for it in range(N_iter+1):
        psi1_t = sim_solutions[it]
        psi2_t = sim_solutions[(N_iter+1) + it]
        
        prob = np.abs(psi1_t)**2 + np.abs(psi2_t)**2
        
        grid.point_data["Psi"][:] = scale_factor * prob
        
        plotter.write_frame()
    
    plotter.close()
    
    return None


def initial_condition(x, state_pop):
    px = p_vec[0]
    py = p_vec[1]
    x0 = initial_coord[0]
    y0 = initial_coord[1]
    
    consts = state_pop / ( gauss_width * np.sqrt(np.pi))
    gauss_pack = np.exp(- ( (x[0] - x0)**2 \
       + (x[1] - y0)**2 ) / (2 * gauss_width ** 2) )
    
    wave_prop = np.exp( 1j * ( px * (x[0] - x0) \
                            + py * (x[1] - y0) ))
    
    return consts * gauss_pack * wave_prop


def CAP_func(x): # Complex absorbing potential
    abs_x = np.abs(x[0]) # - a to optimise
    abs_y = np.abs(x[1]) # - b to optimise
    n = 4
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
    # p1 = np.conj(psi1_arr) * g * ( psi1_arr + psi2_arr ) 
    # p2 = np.conj(psi2_arr) * g * ( psi1_arr + psi2_arr )
    # return p1 + p2
    
    
    # tmp = psi1_arr + psi2_arr
    # return g * np.conj(tmp) * tmp
    
    f1 = np.conj(psi1_arr) * (g * psi1_arr + g3 * psi2_arr)
    f2 = np.conj(psi2_arr) * (g3 * psi1_arr + g * psi2_arr)
    return f1 + f2


def L2_norm_printer(f1, f2):
    # takes two fem.Function arguments
    comm = f1.function_space.mesh.comm
    PSI_norm = fem.form(( f1 * ufl.conj( f1 ) + f2 * ufl.conj( f2 ))* ufl.dx)
    PSI_norm_glob = np.sqrt(comm.allreduce(
        fem.assemble_scalar(PSI_norm), 
        MPI.SUM)
        )
    
    
    if MPI.COMM_WORLD.rank == 0:
        print(PSI_norm_glob)
    
    return None


def aux_func_constructor(domain_mesh):
    # coeff func space; one-to-one connection cell and value
    Q = fem.functionspace(domain_mesh, ("DG", 0)) 
    
    # DEFINE COEFF
    CAP_W = fem.Function(Q)
    CAP_W.name = "CAP = W"
    CAP_W.interpolate(CAP_func)
    
    pot_V = fem.Function(Q)
    pot_V.name = "V"
    pot_V.interpolate(potential_func)
    
    return CAP_W, pot_V


def BC_constructor(domain_mesh, V):
    tdim = domain_mesh.topology.dim # give dim of cells
    fdim = tdim - 1 # dim of facets/edges
    domain_mesh.topology.create_connectivity(fdim, tdim) # edge to cell
    boundary_facets = mesh.exterior_facet_indices(domain_mesh.topology) 
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)
    return bc



def problem_solver(domain_mesh, time_iter_var : list,):
    
    assert np.dtype(PETSc.ScalarType).kind == "c"
    j_im = PETSc.ScalarType(0 + 1j)
    
    
    # UNPACK variables for time stepper
    # T = time_iter_var[0]
    N_iter = time_iter_var[1]
    # dt = time_iter_var[2]
    dt_inv = time_iter_var[3]
    
    
    # DISCRETE SPACES
    # function space
    V = fem.functionspace(domain_mesh, ("Lagrange", 1))
    
    # construct functions
    CAP_W, pot_V = aux_func_constructor(domain_mesh)
    
    # APPLY BC
    bc = BC_constructor(domain_mesh, V)
    bc_nested = [bc, bc]
    
    
    # constructs convection vectors
    b1_vec = ufl.as_vector([1.0 + 0.0j, -1.0j, 0.0j])
    b2_vec = ufl.as_vector([1.0 + 0.0j, 1.0j, 0.0j]) 
    
    
    # DEFINE functions
    # Psi1, Psi2 = ufl.TrialFunction(V), ufl.TrialFunction(V)
    Phi1, Phi2 = ufl.TrialFunction(V), ufl.TrialFunction(V)
    v1, v2 = ufl.TestFunction(V), ufl.TestFunction(V)
    v1_conj = ufl.conj(v1)
    v2_conj = ufl.conj(v2)
    
    # takes output from solver
    psi1_new = fem.Function(V)
    psi2_new = fem.Function(V)
    
    # the last step
    psi1_old = fem.Function(V)
    psi2_old = fem.Function(V)
    
    # linearised non-linear term
    gamma_new = fem.Function(V)
    
    
    # INITIALISE FUNCTIONS
    psi1_old.interpolate(lambda x : initial_condition(x, state_pop1))
    psi2_old.interpolate(lambda x : initial_condition(x, state_pop2))
    gamma_new.x.array[:] = F_nonlin_func(psi1_old.x.array, psi2_old.x.array)
    
    
    #######
    t_start = perf_counter()
    #######
    
    
    # VARIATIONAL FORMS
# =============================================================================
#     # bilinear form         # psi variation
#     a11 = j_im * 2 * dt_inv * Psi1 * v1_conj * ufl.dx
#     a11 += - 0.5 * ufl.inner(ufl.grad(Psi1), ufl.grad(v1)) * ufl.dx 
#     a11 += - pot_V * Psi1 * v1_conj * ufl.dx \
#         + j_im * CAP_W * Psi1 * v1_conj * ufl.dx
#     a11 += - gamma_new * Psi1 * v1_conj * ufl.dx
#     
#     a12 = j_im * u * ufl.dot(b1_vec, ufl.grad(Psi2)) * v1_conj * ufl.dx
#     a21 = j_im * u * ufl.dot(b2_vec, ufl.grad(Psi1)) * v2_conj * ufl.dx
#     # a12 = None
#     # a21 = None
#     
#     a22 = j_im * 2 * dt_inv * Psi2 * v2_conj * ufl.dx
#     a22 += - 0.5 * ufl.inner(ufl.grad(Psi2), ufl.grad(v2)) * ufl.dx 
#     a22 += - pot_V * Psi2 * v2_conj * ufl.dx \
#         + j_im * CAP_W * Psi2 * v2_conj * ufl.dx
#     a22 += - gamma_new * Psi2 * v2_conj * ufl.dx
#     
#     
#     # linear form           # psi variation
#     L1 = j_im * 2 * dt_inv * psi1_old * v1_conj * ufl.dx
#     L1 += 0.5 * ufl.inner(ufl.grad(psi1_old), ufl.grad(v1)) * ufl.dx 
#     L1 += pot_V * psi1_old * v1_conj * ufl.dx \
#         - j_im * CAP_W * psi1_old * v1_conj * ufl.dx
#     L1 += gamma_new * psi1_old * v1_conj * ufl.dx
#     L1 += - j_im * u * ufl.dot(b1_vec, ufl.grad(psi2_old)) * v1_conj * ufl.dx
#     
#     L2 = j_im * 2 * dt_inv * psi2_old * v2_conj * ufl.dx
#     L2 += 0.5 * ufl.inner(ufl.grad(psi2_old), ufl.grad(v2)) * ufl.dx 
#     L2 += pot_V * psi2_old * v2_conj * ufl.dx \
#         - j_im * CAP_W * psi2_old * v2_conj * ufl.dx
#     L2 += gamma_new * psi2_old * v2_conj * ufl.dx
#     L2 += - j_im * u * ufl.dot(b2_vec, ufl.grad(psi1_old)) * v2_conj * ufl.dx
# =============================================================================
    

    # bilinear forms      # phi variation
    a11 = j_im * 2 * dt_inv * Phi1 * v1_conj * ufl.dx
    a11 += - 0.5 * ufl.inner(ufl.grad(Phi1), ufl.grad(v1)) * ufl.dx 
    a11 += - pot_V * Phi1 * v1_conj * ufl.dx \
        + j_im * CAP_W * Phi1 * v1_conj * ufl.dx
    a11 += - gamma_new * Phi1 * v1_conj * ufl.dx
    
    a12 = j_im * u * ufl.dot(b1_vec, ufl.grad(Phi2)) * v1_conj * ufl.dx
    a21 = j_im * u * ufl.dot(b2_vec, ufl.grad(Phi1)) * v2_conj * ufl.dx
    # a12 = None
    # a21 = None
    
    a22 = j_im * 2 * dt_inv * Phi2 * v2_conj * ufl.dx
    a22 += - 0.5 * ufl.inner(ufl.grad(Phi2), ufl.grad(v2)) * ufl.dx 
    a22 += - pot_V * Phi2 * v2_conj * ufl.dx \
        + j_im * CAP_W * Phi2 * v2_conj * ufl.dx
    a22 += - gamma_new * Phi2 * v2_conj * ufl.dx
    
    
    # linear form         # phi variation
    L1 = j_im * 4 * dt_inv * psi1_old * v1_conj * ufl.dx
    L2 = j_im * 4 * dt_inv * psi2_old * v2_conj * ufl.dx
    
    
    # forms
    a_form = fem.form([
        [a11, a12],
        [a21, a22]
        ])
    
    L_form = fem.form([L1, L2])
    

    # CONSTRUCT MATRIX AND VECTOR
    A_block = fem.petsc.assemble_matrix(a_form, bcs=bc_nested, kind="nest")
    A_block.assemble()
    
    b = fem.petsc.create_vector(
        fem.extract_function_spaces(L_form), 
        kind="nest"
        )
    
    
    # CONSTRUCT PETSc SOLVER
    solver = PETSc.KSP().create(domain_mesh.comm)
    solver.setOperators(A_block)
    
    # set types : for serial runs
    solver.setType(PETSc.KSP.Type.GMRES)
    # solver.getPC().setType(PETSc.PC.Type.ILU)
    # solver.getPC().setType(PETSc.PC.Type.LU)
    
    
    pc = solver.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    pc.setFieldSplitIS(
        ("psi1", A_block.getNestISs()[0][0]),
        ("psi2", A_block.getNestISs()[1][1])
        )

    subksps = pc.getFieldSplitSubKSP()
    for subksp in subksps:
        subksp.setType("preonly")
        subpc = subksp.getPC()
        subpc.setType("ilu")


    # pc.setType("hypre")
    # pc.setHYPREType("boomeramg")
    
    
    # store solution vector
    x = PETSc.Vec().createNest([
        create_vector_wrap(psi1_new.x), 
        create_vector_wrap(psi2_new.x)
        ])
    
    
    
    # STORE SOLUTIONS
    complete_sim_solutions = np.zeros(
        ( 2 * (N_iter+1), psi1_old.x.array.shape[0] ),
        dtype=np.complex128()
        )
    
    # store init
    complete_sim_solutions[0] = psi1_old.x.array
    complete_sim_solutions[N_iter + 1] = psi2_old.x.array

    
    # L2 norm
    L2_norm_printer(psi1_old, psi2_old)
    
    
    for it in range(1, N_iter + 1):
        
        # update load vector
        for ii in range(2):
            b_sub = b.getNestSubVecs()[ii]
            with b_sub.localForm() as b_sub_loc:
                b_sub_loc.set(0.0)
        
        # assemble vector
        fem.petsc.assemble_vector(b, L_form)
        
        
        # get trial func space & get correct bcs for nested bilinear form
        bcs1 = fem.bcs_by_block(
            fem.extract_function_spaces(a_form, 1), 
            bc_nested
            )
        # apply lifting, prepare for to set Dirichlet BC
        fem.petsc.apply_lifting(b, a_form, bcs=bcs1)
        
        # update ghosted values
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE
                )
        
        # set bcs on load vector
        bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L_form), bc_nested)
        fem.petsc.set_bc(b, bcs=bcs0)
        
        
        # solve problem & update ghost values
        solver.solve(b, x)
        psi1_new.x.scatter_forward()
        psi2_new.x.scatter_forward()
        
        
        # update solutions & update ghost values
        psi1_old.x.array[:] = psi1_new.x.array - psi1_old.x.array # phi var
        psi2_old.x.array[:] = psi2_new.x.array - psi2_old.x.array # phi var
        # psi1_old.x.array[:] = psi1_new.x.array # psi var
        # psi2_old.x.array[:] = psi2_new.x.array # psi var
        psi1_old.x.scatter_forward()
        psi2_old.x.scatter_forward()
        
        
        gamma_new.x.array[:] = - gamma_new.x.array \
            + 2 * F_nonlin_func(psi1_old.x.array, psi2_old.x.array)
        gamma_new.x.scatter_forward()
        
        # L2_norm_printer(gamma_new, gamma_new)
        
        # update LHS; Operator Matrix
        A_block.zeroEntries()
        fem.petsc.assemble_matrix(A_block, a_form, bcs=bc_nested)
        A_block.assemble()
        # solver.setOperators(A_block)
        
        
        # store solution
        complete_sim_solutions[it] = psi1_old.x.array
        complete_sim_solutions[(N_iter+1)+it] = psi2_old.x.array

    
    # finilise and destroy; no memory leaks
    A_block.destroy()
    b.destroy()
    solver.destroy()
    
    # L2 norm
    L2_norm_printer(psi1_old, psi2_old)
    
    t_end = perf_counter()
    print(t_end - t_start)
    
    return complete_sim_solutions


# Unpack box domain param
data = np.load("sim_data_files/box_params.npz")
box_width_comp_domain = data["bwcd"]
box_height_comp_domain = data["bhcp"]
a = data["a_axis"]
b = data["b_axis"]
delta = data["d"]
potential_coords = data["pot_coords"]
ext_domain_coord_ranges = data["domain_coord_ranges"]
ext_domain_polygon_vertices = data["domain_polygon_vertices"]
h = data["h_max"] # max mesh size
del data


## EQUATION COEFFICIENT PARAMETERS
u = 2 # velocity of coupled term
g = 0.25 # scattering amplitude 
g1 = g # IF CHANGE VALUE HERE, MUST CHAGNE VALUE IN NON-LINEAR FUNC
g2 = g
g3 = 0.1



y0 = 0
initial_coord = np.array([(-a + potential_coords[0][0]) * 0.5, 0])

# radius of start circ
# start_circ_r = np.abs(-a + potential_coords[0][0]) * 0.5
# ang_of_atac = 0.0 # angle of attack against normal to potential barriers centre

# initial coordinates of wave packet
# initial_coord = np.array([- start_circ_r * np.cos(ang_of_atac), 
#                  start_circ_r * np.sin(ang_of_atac)])

# momentum vector of initial cond
# p_vec = - 20 * initial_coord[:] / np.linalg.norm(initial_coord)
p_vec = - 4 * (initial_coord[:] / np.linalg.norm(initial_coord))
# p_vec = [1, 0] # velocity of initial cond

gauss_width = 1 # arbitrarily chosen # smaller diffuses faster
state_pop1 = 1 / np.sqrt(2) # state population
state_pop2 = - state_pop1



## SIM TIME PARAM
T = 1 # end time
N = 200 # number of time steps from 0th
dt = T / N # time step
dt_inv = N / T
sim_time_iter_var = [T, N, dt, dt_inv]


## LOAD MESH
h = 0.08
ext_domain_mesh, _, _ = polygon_mesh(ext_domain_polygon_vertices, h)
# with io.XDMFFile(MPI.COMM_WORLD, "sim_data_files/mesh.xdmf", "r") as xdmf:
#     # ext_domain_mesh = xdmf.read_mesh(name="Grid")
#     ext_domain_mesh = xdmf.read_mesh()
# xdmf.close()



sim_solutions = problem_solver(ext_domain_mesh, sim_time_iter_var)
np.save("sim_data_files/comp_sol_1", sim_solutions)


plotter_func(ext_domain_mesh, sim_solutions, sim_time_iter_var, 
             scale_factor=10, args=True)








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