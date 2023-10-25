import dolfin as df
from fenicstools import interpolate_nonmatching_mesh_any, StructuredGrid
from itertools import product
import h5py
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze steady ADR")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--Lx", type=float, default=1, help="Lx")
    parser.add_argument("--Ly", type=float, default=1, help="Ly")
    parser.add_argument("--Lz", type=float, default=1, help="Lz")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    parser.add_argument("--direction", type=str, default='z', help="x or z direction of flow")
    return parser.parse_args()

    
class Boundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
mpi_root = df.MPI.rank(df.MPI.comm_world) == 0     


if __name__ == "__main__":

    args = parse_args()

    direction = args.direction
    D=args.D
    pore_size = args.L
    it = args.it
    Lx = args.Lx
    Ly = args.Ly
    Lz = args.Lz
    eps = 1e-8

    Pe__ = pore_size / D


    if direction == 'z':
        L = [Lx, Ly, Lz - 2*eps]
        N = [800, 800, 50]
        dx = [Li/Ni for Li, Ni in zip(L, N)]
        origin = [eps for dxi in dx]

        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        dL = [Li-dxi for Li, dxi in zip(L, dx)]
        dL[2] += dx[2]
        N[2] += 1

    if direction == 'x':
        L = [Lx- 2*eps, Ly, Lz]
        N = [50, 800, 800]
        dx = [Li/Ni for Li, Ni in zip(L, N)]
        origin = [eps for dxi in dx]

        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        dL = [Li-dxi for Li, dxi in zip(L, dx)]
        dL[0] += dx[0]
        N[0] += 1


    Jname = ["J_adv", "J_diff", "Iz"]

    # Create the mesh
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), args.mesh + 'mesh.h5', "r") as h5f_up:
        h5f_up.read(mesh_u, "mesh", False)

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    S_u = df.FunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), args.vel + "v_hdffive.h5", "r") as h5f:
        h5f.read(u_, "u")


    subd = df.MeshFunction("size_t", mesh_u, mesh_u.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)

    #wall = Boundary()
    #wall.mark(subd, 1)

    #uwall = df.DirichletBC(S_u, df.Constant(0.), subd, 1)

    if mpi_root:
        print("Initial projection")
    u_x_ = df.project(u_[0], S_u, solver_type="gmres", preconditioner_type="amg")#, bcs=uwall)
    u_y_ = df.project(u_[1], S_u, solver_type="gmres", preconditioner_type="amg")#, bcs=uwall)
    u_z_ = df.project(u_[2], S_u, solver_type="gmres", preconditioner_type="amg")#, bcs=uwall)
    if mpi_root:
        print("done")
        
    if it == 0: 
        mesh = mesh_u
    else: 
        mesh = df.Mesh()
        with df.HDF5File(mesh.mpi_comm(), args.mesh+"refined_mesh/mesh_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f_up:
            h5f_up.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)

    specii = ["a", "b", "c", 'delta']
    conc_ = dict()
    for species in specii:
        conc_[species] = df.Function(S, name=species)
    
    with df.HDF5File(mesh.mpi_comm(), args.con + "abc/con_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f:
        for species in specii:
            h5f.read(conc_[species], species)
            

    Jz_diff_ = dict()
    Iz_ = dict()
    for species in specii:
        #Jz_loc = [u_proj_z_ * conc_[species], - D * conc_[species].dx(2)]
        #Jz_[species] = [df.project(ufl_expr, S, solver_type="gmres", preconditioner_type="amg")
        #                for ufl_expr in Jz_loc]
        if direction == 'z':
            ufl_expr = - D * conc_[species].dx(2)
        if direction == 'x':
            ufl_expr = - D * conc_[species].dx(0)
            
        ufl_Iz = (conc_[species].dx(1))**2
        Jz_diff_[species] = df.project(ufl_expr, S, solver_type="gmres", preconditioner_type="amg")
        Iz_[species] = df.project(ufl_Iz, S, solver_type="gmres", preconditioner_type="amg")
            
    sg = StructuredGrid(S_u, N, origin, vectors, dL)

    xgrid, ygrid, zgrid = sg.create_coordinate_vectors()
    xgrid = xgrid[0]
    ygrid = ygrid[1]
    zgrid = zgrid[2]

    sg(u_x_)
    ux_data = sg.array()
    sg.probes.clear()

    sg(u_y_)
    uy_data = sg.array()
    sg.probes.clear()

    sg(u_z_)
    uz_data = sg.array()
    sg.probes.clear()

    if mpi_root:
        h5f = h5py.File(args.con+"data/abc_data_Pe{}_it{}.h5".format(Pe__, it), "w")
        h5f.create_dataset("ux", data=ux_data.reshape(N[::-1]))
        h5f.create_dataset("uy", data=uy_data.reshape(N[::-1]))
        h5f.create_dataset("uz", data=uz_data.reshape(N[::-1]))
        h5f.create_dataset("x", data=xgrid)
        h5f.create_dataset("y", data=ygrid)
        h5f.create_dataset("z", data=zgrid)  

    sg = StructuredGrid(S, N, origin, vectors, dL)

    for species in specii:
        print(species)
        sg(conc_[species])
        data = sg.probes.array(0)
        if mpi_root:
            h5f.create_dataset("{}/{}".format(species, "conc"), data=data.reshape(N[::-1]))
        sg.probes.clear()

        #for i in range(2):
        #    #sg(Jz_[species][i])
        sg(Jz_diff_[species])
        #    #sg.tovtk(0, filename="dump_scalar_{}_{}.vtk".format(species, i))
        data = sg.probes.array(0)
        if mpi_root:
            h5f.create_dataset("{}/{}".format(species, Jname[1]), data=data.reshape(N[::-1]))
        sg.probes.clear()


        sg(Iz_[species])
        data = sg.probes.array(0)
        if mpi_root:
            h5f.create_dataset("{}/{}".format(species, Jname[2]), data=data.reshape(N[::-1]))
        sg.probes.clear()

    if mpi_root:
        h5f.close()
