/** \file threads_async.cpp
 * \brief Carries out benchmarking tests related to thread-parallel asynchronous preconditioning
 * \author Aditya Kashi
 * \date 2018-03
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include <petscksp.h>

#include "../src/alinalg.hpp"
#include "../src/autilities.hpp"
#include "../src/aoutput.hpp"
#include "../src/aodesolver.hpp"
#include "../src/afactory.hpp"
#include "../src/ameshutils.hpp"

#include <blasted_petsc.h>

using namespace amat;
using namespace acfd;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Carries out benchmarking tests related to thread-parallel \
		asynchronous preconditioning\n\
		Arguments needed: FVENS control file,\n optionally PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);


#ifdef USE_BLASTED
	// write out time taken by BLASTed preconditioner
	if(mpirank == 0) {
		const double linwtime = bctx.factorwalltime + bctx.applywalltime;
		const double linctime = bctx.factorcputime + bctx.applycputime;
		int numthreads = 1;
#ifdef _OPENMP
		numthreads = omp_get_max_threads();
#endif
		std::ofstream outf; outf.open(opts.logfile+"-precon.tlog", std::ofstream::app);
		// if the file is empty, write header
		outf.seekp(0, std::ios::end);
		if(outf.tellp() == 0) {
			outf << "# Time taken by preconditioning operations only:\n";
			outf << std::setw(10) << "# num-cells "
				<< std::setw(6) << "threads " << std::setw(10) << "wall-time "
				<< std::setw(10) << "cpu-time " << std::setw(10) << "avg-lin-iters "
				<< std::setw(10) << " time-steps\n";
		}

		// write current info
		outf << std::setw(10) << m.gnelem() << " "
			<< std::setw(6) << numthreads << " " << std::setw(10) << linwtime << " "
			<< std::setw(10) << linctime
			<< "\n";
		outf.close();
	}
#endif

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}
