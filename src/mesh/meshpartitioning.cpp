/** \file
 * \brief Implementation of mesh partitioning - calls PT-Scotch.
 */

#include "meshpartitioning.hpp"
#include "utilities/mpiutils.hpp"

namespace fvens {

/// Populates this process's share of mesh arrays from the global arrays
static void splitPointsBfaces(
                              const amat::Array2d<a_int>& globalelems,
                              const amat::Array2d<a_int>& globalbface,
                              const amat::Array2d<a_real>& globalcoords,
                              const amat::Array2d<a_int>& inpoel,
                              amat::Array2d<a_int>& bface,
                              amat::Array2d<a_real>& coords);

void partitionMeshTrivial(UMesh2dh<a_real>& m)
{
	StatusCode ierr = 0;
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	if(rank == 0) {
		m.nelemglobal = m.nelem;
		m.npoinglobal = m.npoin;
	}

	MPI_Bcast((void*)&m.nelemglobal, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*)&m.npoinglobal, 1, MPI_INT, 0, MPI_COMM_WORLD);

	const int numloceleminit = m.nelemglobal/nranks;
	const int numlocalelemremain = m.nelemglobal % nranks;
	if(rank == 0)
		m.nelem = numloceleminit + numlocalelemremain;
	else
		m.nelem = numloceleminit;

	// Need to store subdomain-0's local data in a separate array
	amat::Array2d<a_int> sub0inpoel, sub0bface;
	amat::Array2d<a_real> sub0coords;
}

}
