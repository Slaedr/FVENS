/** \file threads_async_tests.cpp
 * \brief Implementation of tests related to multi-threaded asynchronous preconditioning
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

#include "threads_async_tests.hpp"

namespace benchmark {

using namespace acfd;

StatusCode parse_petsc_cmd_options()
{
	StatusCode ierr = 0;
	return ierr;
}

StatusCode test_speedup_sweeps(const FlowParseOptions& opts, const int numthreads,
		const std::vector<int>& sweep_seq, std::ofstream& outfile)
{
	StatusCode ierr = 0;
	return ierr;
}

}

