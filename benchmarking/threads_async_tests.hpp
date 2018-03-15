/** \file threads_async_tests.hpp
 * \brief Declarations of tests related to multi-threaded asynchronous preconditioning
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef THREADS_ASYNC_TESTS_H
#define THREADS_ASYNC_TESTS_H

#include <vector>
#include <string>
#include <fstream>
#include <petscksp.h>

#include "../src/autilities.hpp"

namespace benchmark {

using namespace acfd;

StatusCode parse_petsc_cmd_options();

StatusCode test_speedup_sweeps(const FlowParseOptions& opts, const int numthreads,
		const std::vector<int>& sweep_seq, std::ofstream& outfile);

}

#endif
