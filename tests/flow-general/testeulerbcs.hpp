/** \file testeulerbcs.hpp
 * \brief Tests for general BCs required for either both Euler and Navier-Stokes equations or
 *   those required for only Euler equations.
 */

#ifndef FVENS_TEST_EULERBCS_H
#define FVENS_TEST_EULERBCS_H

#include "spatial/abc.hpp"

namespace fvens { namespace fvens_tests {

/// Test computation of ghost state for the subsonic inflow BC
/** Compares the outgoing Riemann invariant and stagnation speed of sound for ghost state and
 * interior state.
 */
int test_subsonicInflowBC();

}}
