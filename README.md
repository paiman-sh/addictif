# project_rock_mixing
We analyze the mixing process in porous rocks. We can proceed with the following steps given the mesh of a porous rock:

### Velocity from Stokes
1. To get the stokes velocity field, run ```stokes.py```:
```
mpiexec -np [NUM_PROCESSES] python3 stokes.py --mesh path_to_the_mesh_file/ --vel path_where_velocity_will_be_stored/ --direction x 
```
We assume that the HDF5 file of the mesh is stored in path_to_the_mesh_file/. Depending on the desired flow direction, you can change it.

2. We may normalize the velocity field with the following command:
```
python3 normalize_velocity.py --mesh path_to_the_mesh_file/ --vel path_to_the_velocity_file/
```

### Conservative advection-diffusion
3. Run ```ade_steady.py``` to solve the conservative transport to get the concentration of delta:
```
mpiexec -np [NUM_PROCESSES] python3 ade_steady.py --mesh path_to_the_mesh_file/ --vel path_to_the_normalized_velocity_file/ --con path_where_concentration_will_be_stored/ --direction x --D 0.01 --L 0.2 --refine True --it 0
```
Here, we use the same flow direction as the one we considered to derive the velocity field. We may define the diffusion coefficient ```D``` and the pore size ```L```. The option for ```--refine True ``` is going to refine the mesh of the area with a high gradient of concentration and it is going to store the mesh in a separate file.

### Post-processing
4. Run with ```it = 0, 1, 2, etc.``` to have the concentration with refined mesh at the area with a high gradient of concentration.
5. If we assume we have the reaction ```a + b -> c``` with an infinite Damkholer number, we can derive the concentration of ```a```, ```b```, and ```c``` by running ```post_processing_abc.py``` :
```
mpiexec -np [NUM_PROCESSES] python3 post_processing_abc.py --mesh path_to_the_mesh_file/ --vel path_to_the_normalized_velocity_file/ --con path_where_concentration_will_be_stored/ --direction x --D 0.01 --L 0.2 --it 4
```
Note that we run the post_processing code on the results of the final iteration.
