import dolfin as df
from dolfin import *
import numpy as np
import os
from fenicstools import *
from dolfin import Function
import cppimport
import h5py
import argparse
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Solve steady ADR")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()


class GenSubDomain(df.SubDomain):
    def __init__(self, x_min, x_max, tol=df.DOLFIN_EPS_LARGE):
        self.x_min = x_min
        self.x_max = x_max
        self.tol = tol
        super().__init__()

class Top(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2] > self.x_max[2] - self.tol

class Btm(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2] < self.x_min[2] + self.tol
    
class Boundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class SideWallsX(GenSubDomain):
    def inside(self, x, on_boundary): 
        return on_boundary and bool(
            x[0] < self.x_min[0] + self.tol or x[0] > self.x_max[0] - self.tol)
            
class SideWallsY(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and bool( 
            x[1] < self.x_min[1] + self.tol or x[1] > self.x_max[1] - self.tol)

def mpi_max(x):
    x_max_loc = x.max(axis=0)
    x_max = np.zeros_like(x_max_loc)
    comm.Allreduce(x_max_loc, x_max, op=MPI.MAX)
    return x_max

def mpi_min(x):
    x_min_loc = x.min(axis=0)
    x_min = np.zeros_like(x_min_loc)
    comm.Allreduce(x_min_loc, x_min, op=MPI.MIN)
    return x_min

def mpi_print(*args):
    if rank == 0:
        print(*args)

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


# Code for C++ evaluation of absolute gradients of CG1
helper_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>

class Tet {
public:
  Tet(const dolfin::Cell cell){
    for (dolfin::VertexIterator v(cell); !v.end(); ++v)
    {
        const std::size_t pos = v.pos();
        xx_[pos] = v->x(0);
        yy_[pos] = v->x(1);
        zz_[pos] = v->x(2);
    }

    double j11 = xx_[1]-xx_[0];
    double j12 = yy_[1]-yy_[0];
    double j13 = zz_[1]-zz_[0];
    double j21 = xx_[2]-xx_[0];
    double j22 = yy_[2]-yy_[0];
    double j23 = zz_[2]-zz_[0];
    double j31 = xx_[3]-xx_[0];
    double j32 = yy_[3]-yy_[0];
    double j33 = zz_[3]-zz_[0];

    g2x_ = j22*j33-j23*j32;  g3x_ = j13*j32-j12*j33;  g4x_ = j12*j23-j13*j22;
    g2y_ = j23*j31-j21*j33;  g3y_ = j11*j33-j13*j31;  g4y_ = j13*j21-j11*j23;
    g2z_ = j21*j32-j22*j31;  g3z_ = j12*j31-j11*j32;  g4z_ = j11*j22-j12*j21;
    double det = j11 * g2x_ + j12 * g2y_ + j13 * g2z_;
    double d = 1.0/det;
    g2x_ *= d;  g3x_ *= d;  g4x_ *= d;
    g2y_ *= d;  g3y_ *= d;  g4y_ *= d;
    g2z_ *= d;  g3z_ *= d;  g4z_ *= d;
    g1x_ = -g2x_-g3x_-g4x_;  g1y_ = -g2y_-g3y_-g4y_;  g1z_ = -g2z_-g3z_-g4z_;
  }
  void linearbasis(double r,
                   double s,
                   double t,
                   double u,
                   std::vector<double> &N) const
  {
    N[0] = r;
    N[1] = s;
    N[2] = t;
    N[3] = u;
  }
  void linearderiv(std::vector<double> &Nx,
                   std::vector<double> &Ny,
                   std::vector<double> &Nz) const {
    Nx[0] = g1x_;
    Nx[1] = g2x_;
    Nx[2] = g3x_;
    Nx[3] = g4x_;

    Ny[0] = g1y_;
    Ny[1] = g2y_;
    Ny[2] = g3y_;
    Ny[3] = g4y_;

    Nz[0] = g1z_;
    Nz[1] = g2z_;
    Nz[2] = g3z_;
    Nz[3] = g4z_;
  }
private:
  std::array<double, 4> xx_, yy_, zz_;
  double g1x_, g1y_, g1z_;
  double g2x_, g2y_, g2z_;
  double g3x_, g3y_, g3z_;
  double g4x_, g4y_, g4z_;

  //static constexpr std::array<int, 10> perm_ = {-1, -1, -1, -1, 9, 6, 8, 7, 5, 4};
};

class AbsVecCell : public dolfin::Expression
{
public:

  // Create expression with 1 component
  AbsVecCell() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*u->function_space()->mesh(), cell_index);
    const dolfin::FiniteElement element = *u->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);

    const size_t ncoeff = element.space_dimension()/3;

    std::vector<double> coefficients_(3*ncoeff);

    u->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> N_(ncoeff);
    std::fill(N_.begin(), N_.end(), 1./ncoeff);

    double ux = std::inner_product(N_.begin(), N_.end(), coefficients_.begin(), 0.0);
    double uy = std::inner_product(N_.begin(), N_.end(), &coefficients_[1*ncoeff], 0.0);
    double uz = std::inner_product(N_.begin(), N_.end(), &coefficients_[2*ncoeff], 0.0);

    values[0] = sqrt(ux * ux + uy * uy + uz * uz);
  }
  std::shared_ptr<dolfin::Function> u;
};

class AbsGrad : public dolfin::Expression
{
public:

  // Create expression with 1 component
  AbsGrad() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*a->function_space()->mesh(), cell_index);
    //dolfin_cell.get_cell_data(ufc_cell);
    const dolfin::FiniteElement element = *a->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);
    // const size_t dim = 3; // a->function_space()->mesh()->geometry().dim();
    const size_t ncoeff = 4;
 
    std::vector<double> coefficients_(ncoeff);

    a->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> Nx_(ncoeff);
    std::vector<double> Ny_(ncoeff);
    std::vector<double> Nz_(ncoeff);

    Tet tet(dolfin_cell);
    tet.linearderiv(Nx_, Ny_, Nz_);

    double dadx = std::inner_product(Nx_.begin(), Nx_.end(), coefficients_.begin(), 0.0);
    double dady = std::inner_product(Ny_.begin(), Ny_.end(), coefficients_.begin(), 0.0);
    double dadz = std::inner_product(Nz_.begin(), Nz_.end(), coefficients_.begin(), 0.0);

    values[0] = sqrt(dadx * dadx + dady * dady + dadz * dadz);
  }
  std::shared_ptr<dolfin::Function> a;
};

class CellSize : public dolfin::Expression
{
public:

  // Create expression with 1 component
  CellSize() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*mesh, cell_index);

    values[0] = dolfin_cell.h();
  }
  std::shared_ptr<dolfin::Mesh> mesh;
};

int mark_for_refinement(std::shared_ptr<dolfin::MeshFunction<bool>> cell_marker, std::shared_ptr<dolfin::Function> ind, const double tol) {
  std::shared_ptr<const dolfin::Mesh> mesh = ind->function_space()->mesh();
  const dolfin::FiniteElement& element = *ind->function_space()->element();
  assert(element.space_dimension() == 1);
  cell_marker->set_all(false);
  int num_marked = 0;
  for (std::size_t i = 0; i < mesh->num_cells(); ++i)
  {
    dolfin::Cell dolfin_cell(*mesh, i);
    ufc::cell ufc_cell;

    std::vector<double> coefficients_(1);
    std::vector<double> coordinate_dofs_;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs_);

    ind->restrict(coefficients_.data(), element, dolfin_cell,
                  coordinate_dofs_.data(), ufc_cell);

    // std::cout << coefficients_[0] << std::endl;
    bool flag = coefficients_[0] > tol;
    cell_marker->set_value(i, flag);
    if (flag) ++num_marked;
  }
  return num_marked;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<AbsGrad, std::shared_ptr<AbsGrad>, dolfin::Expression>
    (m, "AbsGrad")
    .def(py::init<>())
    .def_readwrite("a", &AbsGrad::a);
  py::class_<CellSize, std::shared_ptr<CellSize>, dolfin::Expression>
    (m, "CellSize")
    .def(py::init<>())
    .def_readwrite("mesh", &CellSize::mesh);
  py::class_<AbsVecCell, std::shared_ptr<AbsVecCell>, dolfin::Expression>
    (m, "AbsVecCell")
    .def(py::init<>())
    .def_readwrite("u", &AbsVecCell::u);
  m.def("mark_for_refinement", &mark_for_refinement, py::arg("cell_marker"), py::arg("ind"), py::arg("tol"));
}

"""


def mpi_rank():
    return df.MPI.rank(df.MPI.comm_world)


def get_inlet_expr(seg, inlet_it, S, params):
    if seg == 0:
        expr_str = "tanh((sqrt(pow(x[0]-x0, 2)+pow(x[1]-y0, 2))-r0)/(sqrt(2)*eps))"
        #expr_str = "tanh((x[0]-0.5)/(sqrt(2)*eps))"
        #expr_str = "0.5*(tanh((x[0]-0.25)/(sqrt(2)*eps))-tanh((x[0]-0.75)/(sqrt(2)*eps)))"
        x0 = params["x0"]
        y0 = params["y0"]
        r0 = params["r0"]
        eps = params["eps"]

        delta_top_expr = df.Expression(expr_str, x0=x0, y0=y0, r0=r0, eps=eps, degree=2)
        rho_top_expr = df.Expression("1.0", degree=2)
        
    else:
        folder = params["folder"]
        Lz = params["Lz"]
        abc2dname = "{}/abc2d_seg{}_it{}_show.h5".format(folder, seg-1, inlet_it)
        with h5py.File(abc2dname, "r") as h5f:
            nodes = np.array(h5f["Mesh/0/mesh/geometry"])
            faces = np.array(h5f["Mesh/0/mesh/topology"])
            a2d = np.array(h5f["VisualisationVector/0"])[:, 0]
            b2d = np.array(h5f["VisualisationVector/1"])[:, 0]
            

        delta_top_expr = InletFunction(nodes, faces, a2d-b2d, Lz)
        rho_top_expr = InletFunction(nodes, faces, a2d+b2d, Lz)
        
    delta_top = df.interpolate(delta_top_expr, S)
    rho_top = df.interpolate(rho_top_expr, S)
    
    return delta_top, rho_top

def mpi_sum(data):
    data = comm.gather(data)
    if mpi_rank() == 0:
        data = sum(data)
    else:
        data = 0
    return data

compiled_fem_module = cppimport.imp('fenicstools.fem.interpolation')

def interpolate_nonmatching_mesh(u0, V):
    """Interpolate from GenericFunction u0 to FunctionSpace V.

    The FunctionSpace V can have a different mesh than that of u0, if u0
    has a mesh.

    """
    u = Function(V)
    compiled_fem_module.interpolate(u0, u)
    return u

helpers = df.compile_cpp_code(helper_code)


df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True

if __name__ == "__main__":
    
    args = parse_args()

    inlet_it=None
    iterative=True
    full_solver=True
    Lz=1.0
    D=args.D  # diffusion coefficient
    pore_size = args.L
    it = args.it 
    eps=0.002
    refine_tol=0.2
    rel_tol = 1e-9

    linear_solver = "bicgstab"
    preconditioner = "hypre_euclid"

    # Create the mesh
    mesh_u = Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), args.mesh+'mesh.h5', "r") as h5f_up:
        h5f_up.read(mesh_u, "mesh", False)

    Pe__ = pore_size / D
    mpi_print("Pe = {}".format(Pe__))

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), args.vel+"v_hdffive_normalized.h5", "r") as h5f:
        h5f.read(u_, "u")
   
    if it == 0: 
      mesh = mesh_u
    else: 
      mesh = Mesh()
      with df.HDF5File(mesh.mpi_comm(), args.mesh+"refined_mesh/mesh_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f_up:
          h5f_up.read(mesh, "mesh", False)
   
    tol = 1e-2

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print(x_max, x_min)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)

    grains = Boundary()
    sidewalls_x = SideWallsX(x_min, x_max, tol)
    sidewalls_y = SideWallsY(x_min, x_max, tol)
    top = Top(x_min, x_max, tol)
    btm = Btm(x_min, x_max, tol)

    grains.mark(subd, 3)
    sidewalls_x.mark(subd, 4)
    sidewalls_y.mark(subd, 5)
    top.mark(subd, 1)
    btm.mark(subd, 2)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)

    xi = df.TrialFunction(S)
    psi = df.TestFunction(S)
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)
    n = df.FacetNormal(mesh)

    expr_str = "tanh((x[1]-0.5)/(sqrt(2)*eps))"
    delta_top_expr = df.Expression(expr_str, eps=0.01, degree=2)
    rho_top_expr = df.Expression("1.0", degree=2)

    delta_top = df.interpolate(delta_top_expr, S)
    rho_top = df.interpolate(rho_top_expr, S)

    mpi_print("interpolating velocity field...")
    #u_proj_ = interpolate_nonmatching_mesh_any(u_, V)
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_)
    mpi_print("done.")

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(args.vel + 'velocity_proj', "u_proj.xdmf")) as xdmff:
        xdmff.write(u_proj_)

    mpi_print("interpolating norm")
    u_norm_ = df.interpolate(df.CompiledExpression(helpers.AbsVecCell(), u=u_proj_, degree=0), S_DG0)
    u_norm_.rename("u_norm", "u_norm")

    mpi_print("interpolating cell size")
    h_ = df.interpolate(df.CompiledExpression(helpers.CellSize(), mesh=mesh, degree=0), S_DG0)
    h_.rename("h", "h")


    mpi_print("computing grid Peclet")
    Pe_el_ = df.Function(S_DG0, name="Pe_el")
    Pe_el_.vector()[:] = u_norm_.vector()[:] * h_.vector()[:] / (2 * D)

    mpi_print("computing tau")
    tau_ = df.Function(S_DG0, name="tau")
    tau_.vector()[:] = h_.vector()[:] / (2 * u_norm_.vector()[:] + 1e-16)
    arr = 1. - 1. / Pe_el_.vector()[:]
    arr[arr < 0] = 0.
    #arr = 1./np.tanh(Pe_el_.vector()[:]) - 1./Pe_el_.vector()[:]
    tau_.vector()[:] *= arr
    mpi_print("done")

    indicator_ = df.Function(S_DG0, name="indicator")


    r_xi = df.dot(u_proj_, df.grad(xi)) - D * df.div(df.grad(xi))
    a_xi = df.dot(u_proj_, df.grad(xi)) * psi * df.dx \
        + D * df.dot(df.grad(psi), df.grad(xi)) * df.dx
    a_xi += tau_ * r_xi * df.dot(u_proj_, df.grad(psi)) * df.dx
    #a_xi += -D * df.dot(n, df.grad(xi)) * psi * ds(3)

    q_delta = df.Constant(0.)
    L_delta = q_delta * psi * df.dx

    delta_ = df.Function(S, name="delta")

    bc_delta_top = df.DirichletBC(S, delta_top, subd, 1)
    bcs_delta = [bc_delta_top]

    problem_delta = df.LinearVariationalProblem(a_xi,
                                                L_delta,
                                                delta_,
                                                bcs=bcs_delta)
    solver_delta = df.LinearVariationalSolver(problem_delta)


    solver_delta.parameters["linear_solver"] = linear_solver
    solver_delta.parameters["preconditioner"] = preconditioner
    solver_delta.parameters["krylov_solver"]["monitor_convergence"] = True
    solver_delta.parameters["krylov_solver"]["relative_tolerance"] = 1e-12
    solver_delta.solve()
    
    a_b = df.Function(S, name="delta")
    a_b.vector()[:] = delta_.vector()[:]


    a_ = df.Function(S, name= "a")
    b_ = df.Function(S, name="b") 
    c_ = df.Function(S, name="c") 

    a_.vector()[:] = np.maximum(0, delta_.vector()[:])
    b_.vector()[:] = np.maximum(0, -delta_.vector()[:])
    c_.vector()[:] = (1 - abs(delta_.vector()[:]))/2

    mpi_print("computing absgrad")
    absgrad_a_b = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=a_b, degree=0), S_DG0)
    #absgrad_a_ = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=a_, degree=0), S_DG0)
    #absgrad_b_ = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=b_, degree=0), S_DG0)
    #absgrad_c_ = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=c_, degree=0), S_DG0)

    indicator_.vector()[:] = h_.vector()[:] * absgrad_a_b.vector()[:]

    with df.XDMFFile(mesh.mpi_comm(), args.con + "delta/con_show_Pe{}_it{}.xdmf".format(Pe__, it)) as xdmff:
        xdmff.parameters.update({"functions_share_mesh": True,"rewrite_function_mesh": False})

        xdmff.write(a_b, 0.)
        xdmff.write(Pe_el_, 0.)
        xdmff.write(tau_, 0.)
        xdmff.write(u_norm_, 0.)
        xdmff.write(h_, 0.)


    with df.XDMFFile(mesh.mpi_comm(), args.con + "abc/con_show_Pe{}_it{}.xdmf".format(Pe__, it)) as xdmff:
        xdmff.parameters.update({"functions_share_mesh": True,"rewrite_function_mesh": False})

        xdmff.write(a_b, 0.)
        xdmff.write(a_, 0.)
        xdmff.write(b_, 0.)
        xdmff.write(c_, 0.)
        xdmff.write(Pe_el_, 0.)
        xdmff.write(tau_, 0.)
        xdmff.write(u_norm_, 0.)
        xdmff.write(h_, 0.)
      
    with df.HDF5File(mesh.mpi_comm(), args.con + "delta/con_Pe{}_it{}.h5".format(Pe__, it), "w") as h5f:
        h5f.write(a_b, "delta")


    with df.HDF5File(mesh.mpi_comm(), args.con + "abc/con_Pe{}_it{}.h5".format(Pe__, it), "w") as h5f:
        h5f.write(a_b, "delta")
        h5f.write(a_, "a")
        h5f.write(b_, "b")
        h5f.write(c_, "c")
  
    # refinement
    mpi_print("marking for refinement")
    cell_markers = df.MeshFunction("bool", mesh, mesh.topology().dim())
    num_marked = helpers.mark_for_refinement(cell_markers, indicator_.cpp_object(), refine_tol)
    mpi_print("refining")
    new_mesh = df.refine(mesh, cell_markers)
    with df.HDF5File(new_mesh.mpi_comm(), args.mesh + "refined_mesh/mesh_Pe{}_it{}.h5".format(Pe__, it+1), "w") as h5f:
        h5f.write(new_mesh, "mesh")
    mpi_print("done")

    prev_size = mpi_sum(mesh.num_cells())
    new_size = mpi_sum(new_mesh.num_cells())
    num_marked = mpi_sum(num_marked)

    if mpi_rank() == 0:
        mpi_print(
            ("Old mesh size: {}\n"
             "Marked cells:  {}\t({:.3f}%)\n"
             "New mesh size: {}\t({:.3f}x)"   
            ).format(prev_size, num_marked, 
                     float(100*num_marked)/prev_size, new_size, 
                     float(new_size)/prev_size))
