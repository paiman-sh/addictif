import dolfin as df
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Make twisting mesh")
    parser.add_argument("-H", default=2.0, type=float, help="Height")
    parser.add_argument("-N", default=1, type=int, help="Number of rotations")
    parser.add_argument("-M", default=20, type=int, help="Mesh resolution")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mesh = df.UnitCubeMesh(args.M, args.M, int(args.H*args.M))
    x = mesh.coordinates()
    y = np.array(x-0.5)
    x[:, 2] = args.H * y[:, 2]
    theta = args.N * np.pi * y[:, 2]
    x[:, 0] = np.cos(theta) * y[:, 0] + np.sin(theta) * y[:, 1]
    x[:, 1] = -np.sin(theta) * y[:, 0] + np.cos(theta) * y[:, 1]

    filedir = os.path.dirname(__file__)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(filedir, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")