/** @file areconstruction.hpp
 * @brief Classes for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#ifndef _GLIBCXX_VECTOR
#include <vector>
#endif

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#define __ARECONSTRUCTION_H 1

namespace acfd
{

/// Abstract class for variable gradient reconstruction schemes
/** For this, we need ghost cell-centered values of flow variables.
 */
class Reconstruction
{
protected:
	const UTriMesh* m;
	/// Cell centers' x-coords
	const amat::Matrix<double>* xc;
	/// Cell centers' y-coords
	const amat::Matrix<double>* yc;
	/// Ghost cell centers' x-coords
	const amat::Matrix<double>* xcg;
	/// Ghost cell centers' y-coords
	const amat::Matrix<double>* ycg;
	/// Number of converved variables
	int nvars;
	/// Cell-centered flow vaiables
	const amat::Matrix<double>* u;
	/// flow variables at ghost cells
	const amat::Matrix<double>* ug;
	/// Cell-centred x-gradients
	amat::Matrix<double>* dudx;
	/// Cell-centred y-gradients
	amat::Matrix<double>* dudy;

public:
	virtual void setup(const UTriMesh* mesh, const amat::Matrix<double>* unk, const amat::Matrix<double>* unkg, amat::Matrix<double>* gradx, amat::Matrix<double>* grady, 
			const amat::Matrix<double>* _xc, const amat::Matrix<double>* _yc, const amat::Matrix<double>* _xcg, const amat::Matrix<double>* _ycg);
	virtual void compute_gradients() = 0;
};

void Reconstruction::setup(const UTriMesh* mesh, const amat::Matrix<double>* unk, const amat::Matrix<double>* unkg, amat::Matrix<double>* gradx, amat::Matrix<double>* grady, 
		const amat::Matrix<double>* _xc, const amat::Matrix<double>* _yc, const amat::Matrix<double>* _xcg, const amat::Matrix<double>* _ycg)
{
	m = mesh;
	u = unk;
	ug = unkg;
	dudx = gradx;
	dudy = grady;
	xc = _xc;
	yc = _yc;
	xcg = _xcg;
	ycg = _ycg;
	nvars = u->cols();
}

/**
 * @brief Implements linear reconstruction using the Green-Gauss theorem over elements.
 * 
 * The scheme is compact.
 */
class GreenGaussReconstruction : public Reconstruction
{
public:
	void compute_gradients();
};

void GreenGaussReconstruction::compute_gradients()
{
	dudx->zeros(); dudy->zeros();
	
	int iface, idim, ielem, jelem, ivar;
	double areainv1, areainv2;
	std::vector<double> ut(nvars);
	
	for(iface = 0; iface < m->gnbface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		areainv1 = 1.0/m->gjacobians(ielem);
		
		for(ivar = 0; ivar < nvars; ivar++)
		{
			ut[ivar] = (u->get(ielem,ivar) + ug->get(iface,ivar))*0.5*m->ggallfa(iface,2);
			(*dudx)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,0))*areainv1;
			(*dudy)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,1))*areainv1;
		}
	}

	for(iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		jelem = m->gintfac(iface,1);
		areainv1 = 1.0/m->gjacobians(ielem);
		areainv2 = 1.0/m->gjacobians(jelem);
		
		for(ivar = 0; ivar < nvars; ivar++)
		{
			ut[ivar] = (u->get(ielem,ivar) + u->get(jelem,ivar))*0.5*m->ggallfa(iface,2);
			(*dudx)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,0))*areainv1;
			(*dudy)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,1))*areainv1;
			(*dudx)(jelem,ivar) -= (ut[ivar] * m->ggallfa(iface,0))*areainv2;
			(*dudy)(jelem,ivar) -= (ut[ivar] * m->ggallfa(iface,1))*areainv2;
		}
	}
}

/// Class implementing linear weighted least-squares reconstruction
class WeightedLeastSquaresReconstruction : public Reconstruction
{
	amat::Matrix<double> w2x2;
	amat::Matrix<double> w2y2;
	amat::Matrix<double> w2xy;
	amat::Matrix<double> w2xu;
	amat::Matrix<double> w2yu;

public:
	void setup(UTriMesh* mesh, amat::Matrix<double>* unk, amat::Matrix<double>* unkg, amat::Matrix<double>* gradx, amat::Matrix<double>* grady, amat::Matrix<double>* _xc, amat::Matrix<double>* _yc, amat::Matrix<double>* _xcg, amat::Matrix<double>* _ycg);
	void compute_gradients();
};

void WeightedLeastSquaresReconstruction::setup(UTriMesh* mesh, amat::Matrix<double>* unk, amat::Matrix<double>* unkg, amat::Matrix<double>* gradx, amat::Matrix<double>* grady, amat::Matrix<double>* _xc, amat::Matrix<double>* _yc, amat::Matrix<double>* _xcg, amat::Matrix<double>* _ycg)
{
	Reconstruction::setup(mesh, unk, unkg, gradx, grady, _xc, _yc, _xcg, _ycg);

	// compute LHS of least-squares problem
}

void WeightedLeastSquaresReconstruction::compute_gradients()
{
	// compute weighted least-squares gradients
}

} // end namespace
