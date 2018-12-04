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
