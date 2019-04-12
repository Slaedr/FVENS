Next steps
==============

MPI
---
- Read mesh on one rank, partition and distribute
- Interface with Scotch
- Enable local renumbering of cells
- Enable BAIJ format with MPI

3D
--
- Generalize mesh to 3D
- Make physics dimension-agnostic and/or 3D

Residual computation
-----------------------
- Make numerical flux classes compute numerical fluxes for all faces, such that kernels can be inlined
- Similarly, make boundary conditions compute ghost states for all relevant faces
