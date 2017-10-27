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
#include "anumericalflux.hpp"
#include "agradientschemes.hpp"
#include "areconstruction.hpp"
#include "aspatial.hpp"
#include <linearoperator.hpp>

namespace acfd {

/// Returns a new inviscid numerical flux context
InviscidFlux* create_mutable_inviscidflux(const std::string& type, 
		const IdealGasPhysics *const p) ;

/// Returns a new immutable inviscid flux context
const InviscidFlux* create_const_inviscidflux(const std::string& type, 
		const IdealGasPhysics *const p) ;

/// Returns a newly-created gradient computation context
GradientScheme* create_mutable_gradientscheme(const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc) ;

/// Returns a newly-created immutable gradient computation context
const GradientScheme* create_const_gradientscheme(const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc) ;

SolutionReconstruction* create_mutable_reconstruction(const std::string& type,
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc,
		const amat::Array2d<a_real> *const gr, const a_real param);

const SolutionReconstruction* create_const_reconstruction(const std::string& type,
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc,
		const amat::Array2d<a_real> *const gr, const a_real param);

/// Creates the appropriate flow solver class
/** This function is needed to instantiate the appropriate class from the \ref FlowFV template.
 */
Spatial<NVARS>* create_mutable_flowSpatialDiscretization(
	const UMesh2dh *const m,                       ///< Mesh context
	const FlowPhysicsConfig& pconf,                ///< Physical data about the problem
	const FlowNumericsConfig& nconf);              ///< Options controlling the numerical method

/// Generates an immutable spatial discretization for slow problems
/** \sa create_mutable_flowSpatialDiscretization
 */
const Spatial<NVARS>* create_const_flowSpatialDiscretization(
	const UMesh2dh *const m,                       ///< Mesh context
	const FlowPhysicsConfig& pconf,                ///< Physical data about the problem
	const FlowNumericsConfig& nconf);              ///< Options controlling the numerical method

}

#endif
