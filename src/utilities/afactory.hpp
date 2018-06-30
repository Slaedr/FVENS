/** \file afactory.hpp
 * \brief Various factories for generating simulation-related objects
 * \author Aditya Kashi
 * \date 2017 October
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef AFACTORY_H
#define AFACTORY_H

#include <string>

#include "aarray2d.hpp"
#include "spatial/anumericalflux.hpp"
#include "spatial/agradientschemes.hpp"
#include "spatial/areconstruction.hpp"
#include "spatial/aspatial.hpp"

namespace fvens {

/// Returns a new inviscid numerical flux context
InviscidFlux* create_mutable_inviscidflux(const std::string& type, 
		const IdealGasPhysics<a_real> *const p) ;

/// Returns a new immutable inviscid flux context
const InviscidFlux* create_const_inviscidflux(const std::string& type, 
		const IdealGasPhysics<a_real> *const p) ;

/// Returns a newly-created gradient computation context
template <int nvars>
GradientScheme<nvars>* create_mutable_gradientscheme(const std::string& type, 
		const UMesh2dh<a_real> *const m, const amat::Array2d<a_real>& rc) ;

/// Returns a newly-created immutable gradient computation context
template <int nvars>
const GradientScheme<nvars>* create_const_gradientscheme(const std::string& type, 
		const UMesh2dh<a_real> *const m, const amat::Array2d<a_real>& rc) ;

/// Returns a solution reconstruction context
/** Solution reconstruction here means computing the values of the conserved variables at faces from
 * the cell-centred values and cell-centred gradients.
 * \param param A parameter that controls the behaviour of some limiters.
 */
SolutionReconstruction* create_mutable_reconstruction(const std::string& type,
                                                      const UMesh2dh<a_real> *const m,
                                                      const amat::Array2d<a_real>& rc,
                                                      const amat::Array2d<a_real> *const gr,
                                                      const a_real param);

/// Returns an immutable solution reconstruction context \sa create_mutable_reconstruction
const SolutionReconstruction* create_const_reconstruction(const std::string& type,
		const UMesh2dh<a_real> *const m, const amat::Array2d<a_real>& rc,
		const amat::Array2d<a_real> *const gr, const a_real param);

/// Creates the appropriate flow solver class
/** This function is needed to instantiate the appropriate class from the \ref FlowFV template.
 */
FlowFV_base* create_mutable_flowSpatialDiscretization(
	const UMesh2dh<a_real> *const m,               ///< Mesh context
	const FlowPhysicsConfig& pconf,                ///< Physical data about the problem
	const FlowNumericsConfig& nconf);              ///< Options controlling the numerical method

/// Generates an immutable spatial discretization for slow problems
/** \sa create_mutable_flowSpatialDiscretization
 */
const FlowFV_base* create_const_flowSpatialDiscretization(
	const UMesh2dh<a_real> *const m,               ///< Mesh context
	const FlowPhysicsConfig& pconf,                ///< Physical data about the problem
	const FlowNumericsConfig& nconf);              ///< Options controlling the numerical method

}

#endif
