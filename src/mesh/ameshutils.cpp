/** \file ameshutils.cpp
 * \brief Implementation of mesh-related functionality like re-ordering etc.
 */

#include <vector>
#include <iostream>
#include <cstring>
#include <boost/algorithm/string.hpp>
#include "ameshutils.hpp"
#include "meshpartitioning.hpp"
#include "meshordering.hpp"
#include "linalg/alinalg.hpp"
#include "spatial/diffusion.hpp"
#include "utilities/aerrorhandling.hpp"
#include "utilities/mpiutils.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

/// Reorders the mesh cells in a given ordering using PETSc
/** Symmetric premutations only.
 * \warning It is the caller's responsibility to recompute things that are affected by the reordering,
 * such as \ref UMesh2dh::compute_topological.
 *
 * \param ordering The ordering to use - "rcm" is recommended. See the relevant
 * [page](www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatOrderingType.html)
 * in the PETSc manual for the full list.
 * \param sd Spatial discretization to be used to generate a Jacobian matrix
 * \param m The mesh context
 */
template <typename scalar>
static StatusCode reorderMeshPetsc(const char *const ordering, const Spatial<freal,1>& sd,
                                   UMesh<scalar,NDIM>& m);

template <typename scalar>
StatusCode preprocessMesh(UMesh<scalar,NDIM>& m)
{
	int ierr = 0;

	char ordstr[PETSCOPTION_STR_LEN];
	PetscBool flag = PETSC_FALSE;
	CHKERRQ(PetscOptionsGetString(NULL, NULL, "-mesh_reorder", ordstr, PETSCOPTION_STR_LEN, &flag));
	if(flag == PETSC_FALSE) {
		std::cout << "preprocessMesh: No reordering requested.\n";
	}
	else {
		const std::string orderstr = ordstr;

		m.compute_topological();
		m.compute_face_data();

		std::cout << "preprocessMesh: Reording cells in " << ordstr << " ordering.\n";

		if(!strcmp(ordstr,"line")) {
			double threshold;
			ierr = PetscOptionsGetReal(NULL, NULL, "-mesh_anisotropy_threshold", &threshold, &flag);
			CHKERRQ(ierr);
			if(!flag) {
				throw std::runtime_error("No anisotropy threshold given!");
			}
			lineReorder(m,threshold);
		}
		else if(orderstr.find("line") != std::string::npos) {
			// split string at _
			std::vector<std::string> orders;
			boost::split(orders, orderstr, boost::is_any_of("_"));
			if(orders.size() != 2)
				throw std::runtime_error("Invalid ordering!");
			const std::string secondorder = orders[1];

			double threshold;
			ierr = PetscOptionsGetReal(NULL, NULL, "-mesh_anisotropy_threshold", &threshold, &flag);
			CHKERRQ(ierr);
			if(!flag) {
				throw std::runtime_error("No anisotropy threshold given!");
			}

			hybridLineReorder(m, threshold, secondorder.c_str());
		}
		else {
			DiffusionMA<1> sd(&m, 1.0, 0.0,
			                  [](const freal *const r, const freal t,
			                     const freal *const u,
			                     freal *const sourceterm)
			                  { sourceterm[0] = 0; },
			                  "NONE");

			ierr = reorderMeshPetsc(ordstr, sd, m); CHKERRQ(ierr);
		}
	}

	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();

	return ierr;
}

UMesh<freal,NDIM> constructMesh(const std::string mesh_path)
{
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	// Read mesh
	if(mpirank == 0)
		std::cout << "Global mesh:\n";

	UMesh<freal,NDIM> gm(readMesh(mesh_path));

	gm.correctBoundaryFaceOrientation();

	gm.compute_topological();
	if(mpirank == 0)
		std::cout << "**" << std::endl;
	MPI_Barrier(PETSC_COMM_WORLD);

	// Partition
	if(mpirank == 0)
		std::cout << "Distributing the mesh\n";
	TrivialReplicatedGlobalMeshPartitioner p(gm);
	//ScotchRGMPartitioner p(gm);
	p.compute_partition();
	UMesh<freal,NDIM> lm = p.restrictMeshToPartitions();

#ifdef DEBUG
	const int mpisize = get_mpi_size(PETSC_COMM_WORLD);
	if(mpisize == 1) {
		const std::array<bool,8> chk = compareMeshes(gm, lm);
		for(int i = 0; i < 8; i++)
			assert(chk[i]);
	}
#endif

	int ierr = preprocessMesh<freal>(lm); 
	fvens_throw(ierr, "Mesh could not be preprocessed!");

#ifdef DEBUG
	std::cout << " Rank " << mpirank << ":\n\t elems = " << lm.gnelem() << ", faces = " << lm.gnaface()
	          << ",\n\t interior faces = " << lm.gninface() << ", phy boun faces = " << lm.gnbface()
	          << ", conn faces = " << lm.gnConnFace() << ",\n\t vertices = " << lm.gnpoin() << std::endl;
	assert(lm.gnelemglobal() == gm.gnelem());
	for(fint iel = 0; iel < lm.gnelem(); iel++) {
		assert(lm.gglobalElemIndex(iel) >= 0);
		assert(lm.gglobalElemIndex(iel) < lm.gnelemglobal());
	}

	// check global face numbering of connectivity faces
	//assert(p.checkConnFaces(lm));
#endif
	return lm;
}

/* Returns a list of cell indices corresponding to the start of each level.
 * The length of the list is one more than the number of levels.
 */
template <typename scalar>
std::vector<fint> levelSchedule(const UMesh<scalar,NDIM>& m)
{
	// zeroth level starts at cell 0
	std::vector<fint> levels;
	levels.push_back(0);

	fint icell = 0;

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

std::array<bool,8> compareMeshes(const UMesh<freal,NDIM>& m1, const UMesh<freal,NDIM>& m2)
{
	std::array<bool,8> isequal;
	isequal[0] = (m1.gnelem() == m2.gnelem());
	isequal[1] = (m1.gnpoin() == m2.gnpoin());
	isequal[2] = (m1.gnbface() == m2.gnbface());
	isequal[3] = true;
	isequal[4] = true;
	isequal[5] = true;
	isequal[6] = true;
	isequal[7] = true;

	for(fint i = 0; i < m1.gnelem(); i++) {
		if(m1.gnnode(i) != m2.gnnode(i)) {
			isequal[3] = false;
			break;
		}
		if(m1.gnfael(i) != m2.gnfael(i)) {
			isequal[4] = false;
			break;
		}
		for(int j = 0; j < m1.gnnode(i); j++) {
			if(m1.ginpoel(i,j) != m2.ginpoel(i,j)) {
				isequal[5] = false;
				break;
			}
		}
	}
	for(fint i = 0; i < m1.gnpoin(); i++) {
		for(int j = 0; j < NDIM; j++)
			if(std::abs(m1.gcoords(i,j)-m2.gcoords(i,j)) > std::numeric_limits<freal>::epsilon())
			{
				isequal[7] = false;
				break;
			}
	}
	static_assert(NDIM==2, "Only 2D is currently supported!");  // change the hard-coded "2" below before removing this line
	for(fint i = 0; i < m1.gnbface(); i++)
		for(int j = 0; j < 2 +m1.gnbtag(); j++)
			if(m1.gbface(i,j) != m2.gbface(i,j)) {
				isequal[6] = false;
				break;
			}

	return isequal;
}

template <typename scalar>
StatusCode reorderMeshPetsc(const char *const ordering, const Spatial<freal,1>& sd, UMesh<scalar,NDIM>& m)
{
	// The implementation must be changed for the multi-process case
	StatusCode ierr = 0;

	// If the ordering requested is not 'natural', reorder the mesh
	if(std::strcmp(ordering,"natural")) {
		Mat A;
		CHKERRQ(MatCreate(PETSC_COMM_SELF, &A));
		CHKERRQ(MatSetType(A, MATSEQAIJ));
		CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m.gnelem(), m.gnelem()));
		CHKERRQ(setJacobianPreallocation<1>(&m, A));

		Vec u;
		CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, m.gnelem(), &u));
		CHKERRQ(VecSet(u,1.0));

		ierr = sd.assemble_jacobian(u, A); CHKERRQ(ierr);
		CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
		CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

		IS rperm, cperm;
		const PetscInt *rinds, *cinds;
		CHKERRQ(MatGetOrdering(A, ordering, &rperm, &cperm));
		CHKERRQ(ISGetIndices(rperm, &rinds));
		CHKERRQ(ISGetIndices(cperm, &cinds));
		// check for symmetric permutation
		for(fint i = 0; i < m.gnelem(); i++)
			assert(rinds[i] == cinds[i]);

		m.reorder_cells(rinds);

		CHKERRQ(ISRestoreIndices(rperm, &rinds));
		ierr = ISDestroy(&rperm); CHKERRQ(ierr);
		ierr = ISDestroy(&cperm); CHKERRQ(ierr);
		CHKERRQ(MatDestroy(&A));
		CHKERRQ(VecDestroy(&u));
	}
	else {
		std::cout << " reorderMesh: Natural ordering requested; doing nothing." << std::endl;
	}
	return ierr;
}

template StatusCode preprocessMesh(UMesh<freal,NDIM>& m);

// template StatusCode reorderMeshPetsc(const char *const ordering, const Spatial<freal,1>& sd,
//                                      UMesh<freal,NDIM>& m);

template std::vector<fint> levelSchedule(const UMesh<freal,NDIM>& m);

}
