/** \file afactory.cpp
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

#include "afactory.hpp"
#include <blockmatrices.hpp>

namespace acfd {

InviscidFlux* create_mutable_inviscidflux(
		const std::string& type, 
		const IdealGasPhysics *const p) 
{
	InviscidFlux *inviflux = nullptr;

	if(type == "VANLEER") {
		inviflux = new VanLeerFlux(p);
		std::cout << " InviscidFluxFactory: Using Van Leer fluxes." << std::endl;
	}
	else if(type == "ROE")
	{
		inviflux = new RoeFlux(p);
		std::cout << " InviscidFluxFactory: Using Roe fluxes." << std::endl;
	}
	else if(type == "HLL")
	{
		inviflux = new HLLFlux(p);
		std::cout << " InviscidFluxFactory: Using HLL fluxes." << std::endl;
	}
	else if(type == "HLLC")
	{
		inviflux = new HLLCFlux(p);
		std::cout << " InviscidFluxFactory: Using HLLC fluxes." << std::endl;
	}
	else if(type == "LLF")
	{
		inviflux = new LocalLaxFriedrichsFlux(p);
		std::cout << " InviscidFluxFactory: Using LLF fluxes." << std::endl;
	}
	else if(type == "AUSM")
	{
		inviflux = new AUSMFlux(p);
		std::cout << " InviscidFluxFactory: Using AUSM fluxes." << std::endl;
	}
	else if(type == "AUSMPLUS")
	{
		inviflux = new AUSMPlusFlux(p);
		std::cout << " InviscidFluxFactory: Using AUSM+ fluxes." << std::endl;
	}
	else
		std::cout << " InviscidFluxFactory: ! Flux scheme not available!" << std::endl;

	return inviflux;
}

const InviscidFlux* create_const_inviscidflux(
		const std::string& type,
		const IdealGasPhysics *const p) 
{
	return const_cast<const InviscidFlux*>(create_mutable_inviscidflux(type, p));
}

GradientComputation* create_mutable_gradientscheme(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc) 
{
	GradientComputation * gradcomp = nullptr;

	if(type == "LEASTSQUARES")
	{
		gradcomp = new WeightedLeastSquaresGradients<NVARS>(m, rc);
		std::cout << " GradientSchemeFactory: Weighted least-squares typeruction will be used.\n";
	}
	else if(type == "GREENGAUSS")
	{
		gradcomp = new GreenGaussGradients<NVARS>(m, rc);
		std::cout << " GradientSchemeFactory: Green-Gauss typeruction will be used." << std::endl;
	}
	else {
		gradcomp = new ZeroGradients<NVARS>(m, rc);
		std::cout << " GradientSchemeFactory: No gradient computation!" << std::endl;
	}

	return gradcomp;
}

const GradientComputation* create_const_gradientscheme(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc) 
{
	return create_mutable_gradientscheme(type, m, rc);
}

}
