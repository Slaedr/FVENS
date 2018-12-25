/** \file
 * \brief Some utilities for MPI programs
 * \author Aditya Kashi
 */

#include "mpiutils.hpp"
#include <cstdlib>
#include <cstdio>
#include <sys/types.h>
#include <unistd.h>

namespace fvens {

template <>
void mpi_all_reduce(a_real *const arr, const a_int count, MPI_Op op, MPI_Comm comm)
{
	MPI_Allreduce(arr, arr, count, FVENS_MPI_REAL, op, comm);
}

void wait_for_debugger()
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(std::getenv("FVENS_MPI_DEBUG") != NULL && rank == 0)
	{
		volatile int debugger_attached = 0;
		std::fprintf(stderr, "PID %ld waiting for debugger...\n", (long)getpid());
		while(debugger_attached == 0) {  }
	}
	// if(std::getenv("FVENS_MPI_DEBUG"))
	// {
	// 	std::fprintf(stderr, "PID %ld waiting for debugger for 10 seconds\n", (long)getpid());
	// 	sleep(10);
	// }
	MPI_Barrier(MPI_COMM_WORLD);
}

}
