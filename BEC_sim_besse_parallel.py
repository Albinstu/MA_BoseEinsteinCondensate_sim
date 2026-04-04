#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:39:22 2026

@author: stu
"""

import numpy as np
from mpi4py import MPI # for creating mesh
from petsc4py import PETSc
from dolfinx import fem, mesh, default_scalar_type, io
from dolfinx.fem import petsc
from dolfinx.la.petsc import create_vector_wrap
import ufl
# from dolfinx.io import gmsh as gmshio
# import gmsh # mesh
from time import perf_counter



# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# EXTRACT DOMAIN
box_width_comp_domain = None
box_height_comp_domain = None
a = None
b = None
delta = None
potential_coords = None
ext_domain_coord_ranges = None
ext_domain_polygon_vertices = None
h = None




## EQUATION COEFFICIENT PARAMETERS
u = 0 # velocity of coupled term
g = 0.0 # scattering amplitude 
g1 = g # IF CHANGE VALUE HERE, MUST CHAGNE VALUE IN NON-LINEAR FUNC
g2 = g
g3 = g


# radius of start circ
start_circ_r = 0.0
ang_of_atac = 0.0 # angle of attack against normal to potential barriers centre

# initial coordinates of wave packet
# initial_coord = np.array([- start_circ_r * np.cos(ang_of_atac), 
#                  start_circ_r * np.sin(ang_of_atac)])
initial_coord = np.array([- start_circ_r * np.cos(ang_of_atac), 
                 start_circ_r * np.sin(ang_of_atac)])

# momentum vector of initial cond
# p_vec = - 20 * initial_coord[:] / np.linalg.norm(initial_coord)
# p_vec = - 4 * (initial_coord[:] / np.linalg.norm(initial_coord))
p_vec = np.copy(initial_coord)
# p_vec = [1, 0] # velocity of initial cond

gauss_width = 1 # arbitrarily chosen # smaller diffuses faster
state_pop1 = 1 / np.sqrt(2) # state population
state_pop2 = - state_pop1




# =============================================================================
# FUNCTIONS
# =============================================================================
def sim_time_variables_writer(sim_time_iter_var):
    np.save("sim_data_files/sim_time_iter_var.npy", sim_time_iter_var)
    return;


def domain_param_reader():
    
    if rank == 0:
        print("Here load init")
        data = np.load("sim_data_files/box_params.npz")
        print("Here load comp")
    
        # Extract into a dictionary (important: np.load returns a special object)
        payload = {
            "bwcd": data["bwcd"],
            "bhcp": data["bhcp"],
            "a_axis": data["a_axis"],
            "b_axis": data["b_axis"],
            "d": data["d"],
            "pot_coords": data["pot_coords"],
            "domain_coord_ranges": data["domain_coord_ranges"],
            "domain_polygon_vertices": data["domain_polygon_vertices"],
            "h_max": data["h_max"],
        }
    
        del data
    else:
        payload = None
    
    # Broadcast to all processes
    payload = comm.bcast(payload, root=0)
    
    # # Unpack on all ranks
    # box_width_comp_domain = payload["bwcd"]
    # box_height_comp_domain = payload["bhcp"]
    # a = payload["a_axis"]
    # b = payload["b_axis"]
    # delta = payload["d"]
    # potential_coords = payload["pot_coords"]
    # ext_domain_coord_ranges = payload["domain_coord_ranges"]
    # ext_domain_polygon_vertices = payload["domain_polygon_vertices"]
    # h = payload["h_max"]
    # return;

    return payload
    


def initial_condition(x, state_pop):
    # px = mom_vec[0]
    # py = mom_vec[1]
    x0 = initial_coord[0]
    y0 = initial_coord[1]
    px = p_vec[0]
    py = p_vec[1]
    
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
    
    tmp = psi1_arr + psi2_arr
    
    return g * np.conj(tmp) * tmp


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
    
    return;



def gather_vector(comm, rank, vec, N_size):
    local = vec.getArray()
    gathered = comm.gather(local, root=0)
    
    if rank == 0:
        # print(gathered)
        return np.concatenate(gathered)
    else:
        return;


def problem_solver(domain_mesh, time_iter_var : list,):
    
    assert np.dtype(PETSc.ScalarType).kind == "c"

    # CHECK IF ALGEBRA PETSc CAN HANDLE COMPLEX
    if np.issubdtype(PETSc.ScalarType, np.complexfloating):
        j_im = PETSc.ScalarType(0 + 1j)
    else:
        print('Complex Treatment is needed')
    
    
    # UNPACK variables for time stepper
    # T = time_iter_var[0]
    N_iter = time_iter_var[1]
    dt = time_iter_var[2]
    dt_inv = time_iter_var[3]
    
    
    # DISCRETE SPACES
    # function space
    V = fem.functionspace(domain_mesh, ("Lagrange", 1))
    # coeff func space; one-to-one connection cell and value
    Q = fem.functionspace(domain_mesh, ("DG", 0)) 
    
    
    # DEFINE COEFF
    CAP_W = fem.Function(Q)
    CAP_W.name = "CAP = W"
    CAP_W.interpolate(CAP_func)
    
    pot_V = fem.Function(Q)
    pot_V.name = "V"
    pot_V.interpolate(potential_func)
    
    
    # APPLY BC
    tdim = domain_mesh.topology.dim # give dim of cells
    fdim = tdim - 1 # dim of facets/edges
    domain_mesh.topology.create_connectivity(fdim, tdim) # edge to cell
    boundary_facets = mesh.exterior_facet_indices(domain_mesh.topology) 
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)
    
    
    # CONSTRUCT FUNCTIONS
    psi1_old = fem.Function(V)
    psi2_old = fem.Function(V)
    
    psi1_new = fem.Function(V)
    psi2_new = fem.Function(V)
    
    # steppers part of relaxation scheme
    gamma_new = fem.Function(V)
    
    
    # trial & test function
    PHI = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    v_conj = ufl.conj(v) # conjugate needed instead of ufl.inner
    
    
    # constructs convection vectors
    b1_vec = ufl.as_vector([1.0 + 0.0j, -1.0j, 0.0j])
    b2_vec = ufl.as_vector([1.0 + 0.0j, 1.0j, 0.0j]) 
    
    
    # INIT functions
    psi1_old.interpolate(lambda x : initial_condition(x, state_pop1))
    psi2_old.interpolate(lambda x : initial_condition(x, state_pop2))
    gamma_new.x.array[:] = F_nonlin_func(psi1_old.x.array, psi2_old.x.array)
        
    
    # CONSTRUCT BILINEAR AND LINEAR FORMS
    # bilinear
    a_const = (j_im * 2 * dt_inv - pot_V + j_im * CAP_W) * PHI * v_conj * ufl.dx
    a_const += - 0.5 * ufl.inner( ufl.grad(PHI), ufl.grad(v) ) * ufl.dx
    a = - gamma_new * PHI * v_conj * ufl.dx + a_const # full bilinear
    
    # linear    
    L1 = j_im * 4 * dt_inv * psi1_old * v_conj * ufl.dx
    L1 += - j_im * u * ufl.dot(b1_vec, ufl.grad(psi2_old)) * v_conj * ufl.dx
    L2 = j_im * 4 * dt_inv * psi2_old * v_conj * ufl.dx
    L2 += - j_im * u * ufl.dot(b2_vec, ufl.grad(psi1_old)) * v_conj * ufl.dx
    
    
    
    # constructs the object and gathers
    a_form = fem.form(a)
    A_mat = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A_mat.assemble() # each process assemble the matrix
    
    
    # assemble linear form; load vector
    L1_form = fem.form(L1)
    L2_form = fem.form(L2)
    load_vec1 = fem.petsc.create_vector(V)
    load_vec2 = fem.petsc.create_vector(V)
    
    
    # Create PETSc solver
    solver = PETSc.KSP().create(domain_mesh.comm)
    solver.setOperators(A_mat)
    
    # PARALLEL Precond.
    solver.setType("gmres")
    pc = solver.getPC()    
    pc.setType("bjacobi")
    pc.setFactorSolverType("ilu")


    # data storage
    #######################################################    
    comm = domain_mesh.comm
    rank = comm.rank
    
    N_dofs = psi1_old.x.petsc_vec.getSize()
    psi1_glob = gather_vector(comm, rank, psi1_old.x.petsc_vec, N_dofs)
    psi2_glob = gather_vector(comm, rank, psi2_old.x.petsc_vec, N_dofs)
    if rank == 0:
        psi1_data = np.zeros((N_iter+1, N_dofs), dtype=np.complex128)
        psi2_data = np.zeros((N_iter+1, N_dofs), dtype=np.complex128)
        
        psi1_data[0, :] = psi1_glob
        psi2_data[0, :] = psi2_glob
    #######################################################

    
    # L2 norm
    L2_norm_printer(psi1_old, psi2_old)
    t = 0
    for it in range(1, N_iter+1):
        t += dt
        
        # update load vector
        with load_vec1.localForm() as lv1_loc:
            lv1_loc.set(0)
        
        with load_vec2.localForm() as lv2_loc:
            lv2_loc.set(0)
        
        # load_vec1.localForm().set(0)
        # load_vec2.localForm().set(0)
        fem.petsc.assemble_vector(load_vec1, L1_form)
        fem.petsc.assemble_vector(load_vec2, L2_form)
        
        # apply lifting, prepare for to set Dirichlet BC
        fem.petsc.apply_lifting(load_vec1, [a_form], bcs=[[bc]])
        fem.petsc.apply_lifting(load_vec2, [a_form], bcs=[[bc]])
        for b_sub in [load_vec1, load_vec2]:
            # fem.petsc.apply_lifting(b_sub, a_form, bcs=[bc])
            b_sub.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE
                )
            # fem.petsc.set_bc(b_sub, bcs=[bc])
        
        # set bcs on load vector
        fem.petsc.set_bc(load_vec1, bcs=[bc])
        fem.petsc.set_bc(load_vec2, bcs=[bc])
        
        
        # solve problem & update ghost values
        solver.solve(load_vec1, psi1_new.x.petsc_vec)
        solver.solve(load_vec2, psi2_new.x.petsc_vec)
        psi1_new.x.scatter_forward()
        psi2_new.x.scatter_forward()
        
        
        # update solutions & update ghost values
        psi1_old.x.array[:] = psi1_new.x.array - psi1_old.x.array
        psi2_old.x.array[:] = psi2_new.x.array - psi2_old.x.array
        psi1_old.x.scatter_forward()
        psi2_old.x.scatter_forward()
        
        
        # update functions
        gamma_new.x.array[:] = - gamma_new.x.array \
            + 2 * F_nonlin_func(psi1_old.x.array, psi2_old.x.array)
        gamma_new.x.scatter_forward()
        
        
        # update LHS; Operator Matrix
        A_mat.zeroEntries()
        fem.petsc.assemble_matrix(A_mat, a_form, bcs=[bc])
        A_mat.assemble()
        
        
        # store solution
        psi1_glob = gather_vector(comm, rank, psi1_old.x.petsc_vec, N_dofs)
        psi2_glob = gather_vector(comm, rank, psi2_old.x.petsc_vec, N_dofs)
        if rank == 0:
            psi1_data[it, :] = psi1_glob
            psi2_data[it, :] = psi2_glob

    
    # finilise and destroy; no memory leaks
    A_mat.destroy()
    load_vec1.destroy()
    load_vec2.destroy()
    solver.destroy()
    
    
    # L2 norm
    L2_norm_printer(psi1_old, psi2_old)
    

    if rank == 0:
        root_return_solutions = np.array([psi1_data, psi2_data])
    else:
        root_return_solutions = None
    

    return root_return_solutions


def aux_func_constructor(domain_mesh):
    # coeff func space; one-to-one connectison cell and value
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


def new_solver(domain_mesh, time_iter_var : list,):
    
    # if throw error than need complex support/treatment
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
    # psi1_old.x.scatter_forward()
    # psi2_old.x.scatter_forward()
    # gamma_new.x.scatter_forward()
    
    
    #######
# =============================================================================
#     t_start = perf_counter()
# =============================================================================
    #######
    
    
    # VARIATIONAL FORMS
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
    

    # pc = solver.getPC()
    # solver.setTolerances(rtol=1e-10, atol=1e-12)


    # pc.setType("fieldsplit")
    # pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # pc.setFieldSplitIS(
    #     ("psi1", A_block.getNestISs()[0][0]),
    #     ("psi2", A_block.getNestISs()[1][1])
    #     )

    # subksps = pc.getFieldSplitSubKSP()
    # for subksp in subksps:
    #     subksp.setType("preonly")
    #     subpc = subksp.getPC()
    #     subpc.setType("ilu")
    
    
    # store solution vector
    x = PETSc.Vec().createNest([
        create_vector_wrap(psi1_new.x), 
        create_vector_wrap(psi2_new.x)
        ])
    
    
    
    # STORE SOLUTIONS
    # complete_sim_solutions = np.zeros(
    #     ( 2 * (N_iter+1), psi1_old.x.array.shape[0] ),
    #     dtype=np.complex128()
    #     )
    
    # store init
    # complete_sim_solutions[0] = psi1_old.x.array
    # complete_sim_solutions[N_iter + 1] = psi2_old.x.array

    
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
        psi1_old.x.scatter_forward()
        psi2_old.x.scatter_forward()
        
        
        gamma_new.x.array[:] = - gamma_new.x.array \
            + 2 * F_nonlin_func(psi1_old.x.array, psi2_old.x.array)
        gamma_new.x.scatter_forward()
        
        
        # update LHS; Operator Matrix
        A_block.zeroEntries()
        fem.petsc.assemble_matrix(A_block, a_form, bcs=bc_nested)
        A_block.assemble()
        # solver.setOperators(A_block)
        
        
        # store solution
        # complete_sim_solutions[it] = psi1_old.x.array
        # complete_sim_solutions[(N_iter+1)+it] = psi2_old.x.array

    
    # finilise and destroy; no memory leaks
    A_block.destroy()
    b.destroy()
    solver.destroy()
    
    # L2 norm
    L2_norm_printer(psi1_old, psi2_old)
    
# =============================================================================
#     t_end = perf_counter()
#     print(t_end - t_start)
# =============================================================================
    
    # return complete_sim_solutions
    return;



def main():
    pass


payload = domain_param_reader()
# Unpack on all ranks
box_width_comp_domain = payload["bwcd"]
box_height_comp_domain = payload["bhcp"]
a = payload["a_axis"]
b = payload["b_axis"]
delta = payload["d"]
potential_coords = payload["pot_coords"]
ext_domain_coord_ranges = payload["domain_coord_ranges"]
ext_domain_polygon_vertices = payload["domain_polygon_vertices"]
h = payload["h_max"]
del payload

u = 2 # velocity of coupled term
g = 0.25 # scattering amplitude 
g1 = g # IF CHANGE VALUE HERE, MUST CHAGNE VALUE IN NON-LINEAR FUNC
g2 = g
g3 = g


# radius of start circ
start_circ_r = np.abs(-a + potential_coords[0][0]) * 0.5
ang_of_atac = 0.0 # angle of attack against normal to potential barriers centre

# initial coordinates of wave packet
initial_coord = np.array([- start_circ_r * np.cos(ang_of_atac), 
                 start_circ_r * np.sin(ang_of_atac)])

# momentum vector of initial cond
# p_vec = - 20 * initial_coord[:] / np.linalg.norm(initial_coord)
p_vec = - 4 * (initial_coord[:] / np.linalg.norm(initial_coord))
# p_vec = [1, 0] # velocity of initial cond

gauss_width = 1 # arbitrarily chosen # smaller diffuses faster
state_pop1 = 1 / np.sqrt(2) # state population
state_pop2 = - state_pop1



T = 1 # end time
N = 200 # number of time steps from 0th
dt = T / N # time step
dt_inv = N / T
sim_time_iter_var = [T, N, dt, dt_inv] 


# h = 0.09 # max mesh size
# ext_domain_mesh, _, _ = polygon_mesh(ext_domain_polygon_vertices, h)

with io.XDMFFile(MPI.COMM_WORLD, "sim_data_files/mesh.xdmf", "r") as xdmf:
    # ext_domain_mesh = xdmf.read_mesh(name="Grid")
    ext_domain_mesh = xdmf.read_mesh()
xdmf.close()

# complete_sim_solutions = problem_solver(ext_domain_mesh, sim_time_iter_var)
complete_sim_solutions = new_solver(ext_domain_mesh, sim_time_iter_var)
# if ext_domain_mesh.comm.rank == 0:
#     solution_writer(complete_sim_solutions)






# if __name__ == "__main__":
#     main()




# =============================================================================
# WHAT TO DO IN ORDER
# 1. write functions to files
# 2. read mesh from file
# 3. generate mesh in other program
# 4. place constants in file that numpy can read
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