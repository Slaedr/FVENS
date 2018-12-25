/** \file
 * \brief Some convenience functions related to MPI
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MPIUTILS_H
#define FVENS_MPIUTILS_H

#include <mpi.h>
#include "aconstants.hpp"
#include "aerrorhandling.hpp"

namespace fvens {

inline int get_mpi_size(MPI_Comm comm)
{
	int size;
	int ierr = MPI_Comm_size(comm, &size);
	mpi_throw(ierr, "Could not get size!");
	return size;
}

inline int get_mpi_rank(const MPI_Comm comm)
{
	int rank;
	int ierr = MPI_Comm_rank(comm, &rank);
	mpi_throw(ierr, "Could not get rank!");
	return rank;
}

/// Generic wrapper for MPI allreduce from an array into itself
template <typename scalar>
void mpi_all_reduce(scalar *const arr, const a_int count, MPI_Op op, MPI_Comm comm);

/// Waits until a debugger is attached and a variable is changed
/** Only activated if environment variable FVENS_MPI_DEBUG is set.
 * Waits until a variable called 'debugger_attached' is set to 1 using the attached debugger.
 * \note Compiles only on Unix-like systems.
 * Ref: Tom Fogal. "Debugging MPI programs with the GNU debugger". Version 1.1.0. February 2014.
 *   http://www.sci.utah.edu/~tfogal/academic/Fogal-ParallelDebugging.pdf
 */
void wait_for_debugger();

}

#endif
