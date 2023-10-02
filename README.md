# project_rock_mixing
We analyze the mixing process in porous rocks. We can proceed with the following steps given the mesh of a porous rock:

### Velocity from Stokes
1. Depending on the desired direction of flow, run ```stokes_xdir.py``` or ```stokes_zdir.py```:
```
mpiexec -np [NUM_PROCESSES] python3 stokes_xdir.py --mesh path_to_the_mesh_file/ --vel path_where_velocity_will_be_stored/
```
2. We may normalize the velocity field with the following command:
```
python3 normalize_velocity.py --mesh path_to_the_mesh_file/ --vel path_to_the_velocity_file/
```

### Conservative advection-diffusion
3. Run ```ade_xdir.py``` or ```ade_zdir.py``` depending on the desired direction of flow to get the concentration:
```
mpiexec -np [NUM_PROCESSES] python3 ade_xdir.py --mesh path_to_the_mesh_file/ --vel path_to_the_normalized_velocity_file/ --con path_where_concentration_will_be_stored/ --D 0.01 --L 0.2 --it 0
```
4. Run with ```it = 0, 1, 2, etc.``` to have the concentration with refined mesh at the interface.
