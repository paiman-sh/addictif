import dolfin as df
from dolfin import *
import numpy as np
import os
from fenicstools import *
from dolfin import Function
import cppimport
import h5py
import argparse


### we must run it wihtout MPI


def parse_args():
    parser = argparse.ArgumentParser(description="Solve steady ADR")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    return parser.parse_args()


def mpi_rank():
    return df.MPI.rank(df.MPI.comm_world)


def mpi_print(obj):
    if mpi_rank() == 0:
        print(obj)



if __name__ == "__main__":
    
    args = parse_args()
    # Create the mesh
    mesh_u = Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), args.mesh+'mesh.h5', "r") as h5f_up:
        h5f_up.read(mesh_u, "mesh", False)


    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), args.vel+"v_hdffive.h5", "r") as h5f:
        h5f.read(u_, "u")

    
    u_array = np.array(u_.vector()[:])
    u_max = np.max(np.abs(u_array))    
    u_array /= u_max
    u_.vector()[:] = u_array
    
    
    # Create XDMF files for visualization output
    xdmffile_u = XDMFFile(args.vel+'v_normalized_show.xdmf')
    xdmffile_u.parameters["flush_output"] = True
    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_)


    with df.HDF5File(mesh_u.mpi_comm(), args.vel+"v_hdffive_normalized.h5", "w") as h5f:
        h5f.write(u_, "u")

