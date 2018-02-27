/** \file ameshutils.cpp
 * \brief Implementation of mesh-related functionality like re-ordering etc.
 */

#include "ameshutils.hpp"
#include <vector>
#include <iostream>
#include "alinalg.hpp"

namespace acfd {

StatusCode reorderMesh(const char *const ordering, const Spatial<1>& sd, UMesh2dh& m)
{
	// The implementation must be changed for the multi-process case
	
	Mat A;
	CHKERRQ(MatCreate(PETSC_COMM_SELF, &A));
	CHKERRQ(MatSetType(A, MATSEQAIJ));
	CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m.gnelem(), m.gnelem()));
	CHKERRQ(setJacobianPreallocation<1>(&m, A));

	Vec u;
	CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, m.gnelem(), &u));
	CHKERRQ(VecSet(u,1.0));

	CHKERRQ(sd.compute_jacobian(u, A));
	CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
	CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

	IS rperm, cperm;
	const PetscInt *rinds, *cinds;
	CHKERRQ(MatGetOrdering(A, ordering, &rperm, &cperm));
	CHKERRQ(ISGetIndices(rperm, &rinds));
	CHKERRQ(ISGetIndices(cperm, &cinds));
	// check for symmetric permutation
	for(a_int i = 0; i < m.gnelem(); i++)
		assert(rinds[i] == cinds[i]);

	m.reorder_cells(rinds);
	
	CHKERRQ(ISRestoreIndices(rperm, &rinds));
	CHKERRQ(MatDestroy(&A));
	CHKERRQ(VecDestroy(&u));
	return 0;
}

StatusCode preprocessMesh(UMesh2dh& m)
{
	char ordstr[PETSCOPTION_STR_LEN];
	PetscBool flag = PETSC_FALSE;
	CHKERRQ(PetscOptionsGetString(NULL, NULL, "-mesh_reorder", ordstr, PETSCOPTION_STR_LEN, &flag));
	if(flag == PETSC_FALSE) {
		std::cout << "preprocessMesh: No reordering requested.\n";
	}
	else {
		std::cout << "preprocessMesh: Reording cells in " << ordstr << " ordering.\n";
		m.compute_topological();
		m.compute_face_data();

		DiffusionMA<1> sd(&m, 1.0, 0.0, 
			[](const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm)
			{ sourceterm[0] = 0; }, 
		"NONE");

		CHKERRQ(reorderMesh(ordstr, sd, m));
	}
		
	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();

	return 0;
}

/* Returns a list of cell indices corresponding to the start of each level.
 * The length of the list is one more than the number of levels.
 */
std::vector<a_int> levelSchedule(const UMesh2dh& m)
{
	// zeroth level starts at cell 0
	std::vector<a_int> levels;
	levels.push_back(0);
	
	a_int icell = 0;
	
	std::vector<bool> marked(m.gnelem(), false);

	while(icell < m.gnelem()-1)
	{
		// mark current cell
		marked[icell] = true;

		// mark all neighbors
		for(int iface = 0; iface < m.gnfael(icell); iface++) 
		{
			const int othercell = m.gesuel(icell,iface);
			if(othercell < m.gnelem())
				marked[othercell] = true;
		}

		/* If the next cell is among marked cells, this level ends at this cell
		 * and the next level starts at the next cell.
		 */
		if(marked[icell+1]) {
			levels.push_back(icell+1);
			marked.assign(m.gnelem(),false);
		}

		icell++;
	}

	levels.push_back(m.gnelem()); // mark the end of the list

	return levels;
}

}
