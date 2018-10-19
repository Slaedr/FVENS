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

#include "utilities/controlparser.hpp"
#include "linalg/alinalg.hpp"
#include "ode/aodesolver.hpp"

namespace benchmark {

using namespace fvens;

/// Parameters defining the test
/** There is one run for each pair of corresponding entries in the build and apply sweep arrays.
 */
struct SpeedupSweepsConfig {
	std::vector<int> threadSeq;      ///< Sequence of threads to run (apart from base case)
	std::vector<int> buildSwpSeq;    ///< Sequence of build sweeps to run
	std::vector<int> applySwpSeq;    ///< Sequence of apply sweeps to run corresponding to build sweeps
	int basethreads;                 ///< Number of threads to use for base case
	int basebuildsweeps;             ///< No. of build sweeps to use for base case
	int baseapplysweeps;             ///< No. of apply sweeps to use for base case
};

/// Find the dependence of the speed-up from a certain thread-count on the number of async sweeps
/** Only for implicit solves.
 * Runs a nonlinear solve first with 1 thread and 1 sweep (for reference). Then, for each number of
 * threads requested, there's one run for each number of sweeps requested.
 * Each `run' with one of the requested number of threads and one of the requested number of sweeps
 * consists of several repetitions, to account for the asynchronous chaos.
 * If one repetition of a run does not converge, that run is terminated but its details are written
 * to the output file, and it moves on to the next requested number of sweeps.
 *
 * \param opts Numerics and physics options for the test-case
 * \param numrepeat The number of times to repeat the benchmark and average the results
 * \param config Test configuration
 * \param outfile Opened stream context to write the averaged results to
 */
StatusCode test_speedup_sweeps(const FlowParserOptions& opts,
                               const int numrepeat, const SpeedupSweepsConfig config,
                               std::ofstream& outfile);

/// Run a timing test with a specific number of sweeps with a specific number of threads
/** 
 * \param nbswps Number of build sweeps
 * \para, naswps Number of apply sweeps
 * \param bctx BLASTed data context which is re-constructed and thus initialized to zero before use,
 *   and contains timing data of the preconditioner on exit.
 * \return Performance data for the run
 */
TimingData run_sweeps(const Spatial<a_real,NVARS> *const startprob,
                      const Spatial<a_real,NVARS> *const prob,
                      const SteadySolverConfig& maintconf, const int nbswps, const int naswps,
                      KSP *ksp, Vec u, Mat A, Mat M,
                      MatrixFreeSpatialJacobian<NVARS>& mfjac, const PetscBool mf_flg,
                      Blasted_data_list& bctx);

}

#endif
