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

GradientScheme* create_mutable_gradientscheme(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc) 
{
	GradientScheme * gradcomp = nullptr;

	if(type == "LEASTSQUARES")
	{
		gradcomp = new WeightedLeastSquaresGradients<NVARS>(m, rc);
		std::cout << " GradientSchemeFactory: Weighted least-squares gradients will be used.\n";
	}
	else if(type == "GREENGAUSS")
	{
		gradcomp = new GreenGaussGradients<NVARS>(m, rc);
		std::cout << " GradientSchemeFactory: Green-Gauss gradients will be used." << std::endl;
	}
	else {
		gradcomp = new ZeroGradients<NVARS>(m, rc);
		std::cout << " GradientSchemeFactory: No gradient computation!" << std::endl;
	}

	return gradcomp;
}

const GradientScheme* create_const_gradientscheme(
		const std::string& type, 
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc) 
{
	return create_mutable_gradientscheme(type, m, rc);
}

SolutionReconstruction* create_mutable_reconstruction(const std::string& type,
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc,
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
		reconst = new WENOReconstruction(m, rc, gr);
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
		const UMesh2dh *const m, const amat::Array2d<a_real> *const rc,
		const amat::Array2d<a_real> *const gr, const a_real param)
{
	return create_mutable_reconstruction(type, m, rc, gr, param);
}

Spatial<NVARS>* create_mutable_flowSpatialDiscretization(const UMesh2dh *const m, 
		const a_real gamma, const a_real Minf, const a_real Tinf, const a_real Reinf, const a_real Pr, 
		const a_real alpha, const bool viscsim, const bool useconstvisc,
		const int isothermalwall_marker, const int adiabaticwall_marker, 
		const int isothermalpressurewall_marker,
		const int slipwall_marker, const int farfield_marker, const int inout_marker, const int extrap_marker, 
		const int periodic_marker,
		const a_real twalltemp, const a_real twallvel, const a_real adiawallvel, const a_real tpwalltemp, 
		const a_real tpwallvel, const a_real tpwallpressure,
		const std::string invflux, const std::string invfluxjac, const std::string reconst, 
		const std::string limiter, const bool order2, const bool reconstPrim)
{
	if(order2)
		if(useconstvisc)
			return new FlowFV<true,true>(m, gamma, Minf, Tinf, Reinf, Pr, alpha, viscsim, 
				isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
				slipwall_marker, farfield_marker, inout_marker, extrap_marker, periodic_marker,
				twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
				invflux, invfluxjac, reconst, limiter, reconstPrim);
		else
			return new FlowFV<true,false>(m, gamma, Minf, Tinf, Reinf, Pr, alpha, viscsim, 
				isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
				slipwall_marker, farfield_marker, inout_marker, extrap_marker, periodic_marker,
				twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
				invflux, invfluxjac, reconst, limiter, reconstPrim);
	else
		if(useconstvisc)
			return new FlowFV<false,true>(m, gamma, Minf, Tinf, Reinf, Pr, alpha, viscsim, 
				isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
				slipwall_marker, farfield_marker, inout_marker, extrap_marker, periodic_marker,
				twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
				invflux, invfluxjac, reconst, limiter, reconstPrim);
		else
			return new FlowFV<false,false>(m, gamma, Minf, Tinf, Reinf, Pr, alpha, viscsim, 
				isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
				slipwall_marker, farfield_marker, inout_marker, extrap_marker, periodic_marker,
				twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
				invflux, invfluxjac, reconst, limiter, reconstPrim);
}

const Spatial<NVARS>* create_const_flowSpatialDiscretization(const UMesh2dh *const m, 
		const a_real gamma, const a_real Minf, const a_real Tinf, const a_real Reinf, const a_real Pr, 
		const a_real alpha, const bool viscsim, const bool useconstvisc,
		const int isothermalwall_marker, const int adiabaticwall_marker, 
		const int isothermalpressurewall_marker,
		const int slipwall_marker, const int farfield_marker, const int inout_marker, const int extrap_marker, 
		const int periodic_marker,
		const a_real twalltemp, const a_real twallvel, const a_real adiawallvel, const a_real tpwalltemp, 
		const a_real tpwallvel, const a_real tpwallpressure,
		const std::string invflux, const std::string invfluxjac, const std::string reconst, 
		const std::string limiter, const bool order2, const bool reconstPrim)
{
	return create_mutable_flowSpatialDiscretization(
		m, gamma, Minf, Tinf, Reinf, Pr, alpha, viscsim, useconstvisc,
		isothermalwall_marker, adiabaticwall_marker, isothermalpressurewall_marker,
		slipwall_marker, farfield_marker, inout_marker, extrap_marker, periodic_marker,
		twalltemp, twallvel, adiawallvel, tpwalltemp, tpwallvel, tpwallpressure,
		invflux, invfluxjac, reconst, limiter, order2, reconstPrim);
}

}
