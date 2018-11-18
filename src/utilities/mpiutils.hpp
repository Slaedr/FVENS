/** \file
 * \brief Some convenience functions related to MPI
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MPIUTILS_H
#define FVENS_MPIUTILS_H

#include "aerrorhandling.hpp"

int get_mpi_size(MPI_Comm comm);
{
	int size;
	int ierr = MPI_Comm_rank(comm, &size);
	mpi_throw(ierr, "Could not get size!");
	return size;
}

int get_mpi_rank(MPI_Comm comm)
{
	int rank;
	int ierr = MPI_Comm_rank(comm, &rank);
	mpi_throw(ierr, "Could not get rank!");
	return rank;
}

#endif
