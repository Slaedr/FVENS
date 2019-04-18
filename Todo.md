Next steps
==============

MPI
---
- Read partitioned meshes from Gmsh 4.x format files
- Enable local renumbering of cells

3D
--
- Generalize mesh to 3D (no need of internal faces - we will only need boundary faces)
- Make physics dimension-agnostic and/or 3D

Residual computation
-----------------------
- Convert all operations to cell-based loops. 
  For face-unique operations: (option 1) visit each face of the cell and check whether the neighboring cell has lower index; if it has lower index, the face's operation must already have been done; (option 2) double-compute face-unique quantities.
- Make boundary conditions compute ghost states for all relevant faces
