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

#include <iostream>
#include "afactory.hpp"

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

template <int nvars>
GradientScheme<nvars>* create_mutable_gradientscheme(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc) 
{
	GradientScheme<nvars> * gradcomp = nullptr;

	if(type == "LEASTSQUARES")
	{
		gradcomp = new WeightedLeastSquaresGradients<nvars>(m, rc);
		std::cout << " GradientSchemeFactory: Weighted least-squares gradients will be used.\n";
	}
	else if(type == "GREENGAUSS")
	{
		gradcomp = new GreenGaussGradients<nvars>(m, rc);
		std::cout << " GradientSchemeFactory: Green-Gauss gradients will be used.\n";
	}
	else {
		gradcomp = new ZeroGradients<nvars>(m, rc);
		std::cout << " GradientSchemeFactory: No gradient computation.\n";
	}

	return gradcomp;
}

template <int nvars>
const GradientScheme<nvars>* create_const_gradientscheme(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc) 
{
	return create_mutable_gradientscheme<nvars>(type, m, rc);
}

// template instantiations
template GradientScheme<NVARS>* create_mutable_gradientscheme<NVARS>(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc);

template const GradientScheme<NVARS>* create_const_gradientscheme<NVARS>(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc);

template GradientScheme<1>* create_mutable_gradientscheme<1>(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc);

template const GradientScheme<1>* create_const_gradientscheme<1>(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc);


SolutionReconstruction* create_mutable_reconstruction(const std::string& type,
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc,
		const amat::Array2d<a_real> *const gr, const a_real param)
{
	SolutionReconstruction * reconst = nullptr;

	if(type == "NONE")
	{
		reconst = new LinearUnlimitedReconstruction(m, rc, gr);
		std::cout << " ReconstructionFactory: Unlimited linear reconstruction selected.\n";
	}
	else if(type == "WENO")
	{
		reconst = new WENOReconstruction(m, rc, gr, param);
		std::cout << " ReconstructionFactory: WENO reconstruction selected.\n";
	}
	else if(type == "VANALBADA")
	{
		reconst = new MUSCLVanAlbada(m, rc, gr);
		std::cout << " ReconstructionFactory: Van Albada MUSCL reconstruction selected.\n";
	}
	else if(type == "BARTHJESPERSEN")
	{
		reconst = new BarthJespersenLimiter(m, rc, gr);
		std::cout << " ReconstructionFactory: Barth-Jespersen linear reconstruction selected.\n";
	}
	else if(type == "VENKATAKRISHNAN")
	{
		reconst = new VenkatakrishnanLimiter(m, rc, gr, param);
		std::cout << " ReconstructionFactory: Venkatakrishnan linear reconstruction selected.\n";
	}
	else {
		std::cout << " !ReconstructionFactory: Invalid reconstruction!!\n";
	}

	return reconst;
}

const SolutionReconstruction* create_const_reconstruction(const std::string& type,
		const UMesh2dh *const m, const amat::Array2d<a_real>& rc,
		const amat::Array2d<a_real> *const gr, const a_real param)
{
	return create_mutable_reconstruction(type, m, rc, gr, param);
}

FlowFV_base* create_mutable_flowSpatialDiscretization(
	const UMesh2dh *const m,
	const FlowPhysicsConfig& pconf,
	const FlowNumericsConfig& nconf)
{
	if(nconf.order2)
		if(pconf.const_visc)
			return new FlowFV<true,true>(m, pconf, nconf);
		else
			return new FlowFV<true,false>(m, pconf, nconf);
	else
		if(pconf.const_visc)
			return new FlowFV<false,true>(m, pconf, nconf);
		else
			return new FlowFV<false,false>(m, pconf, nconf);
}

const FlowFV_base* create_const_flowSpatialDiscretization(
	const UMesh2dh *const m,
	const FlowPhysicsConfig& pconf,
	const FlowNumericsConfig& nconf)
{
	return create_mutable_flowSpatialDiscretization(m, pconf, nconf);
}

}
