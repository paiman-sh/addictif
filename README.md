 # AdDiCTIF
This repository contains an implementation of *Ad*vection-*Di*ffusion-*C*hemistry in a *T*ime-*I*ndependent *F*ramework.
The methodology is described in [our paper](https://www.sciencedirect.com/science/article/pii/S0309170824001787). In particular, the workflow relies on the finite element method (via FEniCS) to solve steady advection-diffusion equations in arbitrary porous geometries in order to assess transverse mixing.
Assuming instantaneous reaction kinetics we can then, by simple post-processing, infer the concentration of chemical species entering into various chemical reaction networks, allowing the estimation of the relevant reaction rates. We also supply several scripts to analyze the conservative and reactive dynamics.

### Installation
To install, execute:
```
pip install .
```

We can proceed with the following steps given a mesh:

### Velocity from Stokes
1. To get the Stokes velocity field, run ```addictif stokes```:
```
mpiexec -np [NUM_PROCESSES] addictif stokes --mesh path_to_the/mesh_file.h5 -o path_to_the/output_folder/ [--direction [x/y/*z*]] [--tol [tolerance]] [--sidewall_bc [noslip/*freeslip*]]
```
We assume that the HDF5 file of the mesh is stored in path_to_the_mesh_file/. Depending on the desired flow direction, you can change it.

### Conservative advection-diffusion
2. Run ```addictif ade_steady``` to solve the conservative transport to get the concentration of delta:
```
mpiexec -np [NUM_PROCESSES] addictif ade_steady -i path_to_the/output_folder/ -D [diffusion coefficient] --it [iteration 0/1/...]
```
We may define the diffusion coefficient ```D``` and the iteration number. The latter should be 0 at the first run.

3. Run ```addictif refine``` to refine given a conservative field.
```
mpiexec -np [NUM_PROCESSES] addictif refine -i  path_to_the/mesh_folder_at_iteration_x/ --it x
```
4. Then go back to 2 and increase `it` by x+1. Run with ```it = 0, 1, 2, etc.``` to have the concentration with refined mesh at the area with a high gradient of concentration.

### Post-processing
5. If we assume we have the reaction ```a + b -> c``` with an infinite Damkholer number, we can derive the concentration of ```a```, ```b```, and ```c``` by running ```post_processing_abc.py``` :
```
mpiexec -np [NUM_PROCESSES] addictif postprocess_abc -i path_to_the/output_folder/
```
Note that we run the post_processing code on the results of the final iteration.
6.  We can interpolate the solution into a Cartesian grid. This step enables easy analysis of different cross-sections by running ```analyze_data.py``` :
```
mpiexec -np [NUM_PROCESSES] addictif analyze_data -i path_to_the/output_folder/ -Nt [Number of nodes based on desired resolution]
```
7.  we can compute the averages of different indicators in each cross-section by running ```compute_averages.py``` :
```
addictif compute_averages -i path_to_intpdata.h5
```

