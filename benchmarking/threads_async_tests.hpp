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

#include "../src/autilities.hpp"

namespace benchmark {

using namespace acfd;

/// Find the dependence of the speed-up from a certain thread-count on the number of async sweeps
StatusCode test_speedup_sweeps(const FlowParseOptions& opts, const int numthreads,
		const std::vector<int>& sweep_seq, std::ofstream& outfile);

StatusCode run_sweeps(const Spatial<NVARS> *const startprob, const Spatial<NVARS> *const prob,
		const SteadySolverConfig& maintconf, const int nswps,
		KSP ksp, Vec u, Mat A, Mat M, Blasted_data& bctx);

}

#endif
