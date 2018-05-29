#! /bin/bash

#   This file is part of FVENS.
#    FVENS is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    FVENS is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with FVENS.  If not, see <http://www.gnu.org/licenses/>.

# Build the meshes needed

for imesh in `seq 1 3`; do
	gmsh -setnumber refine $imesh -2 -o dom-$imesh.msh dom.geo
done
	
