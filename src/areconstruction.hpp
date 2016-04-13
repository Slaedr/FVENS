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

#ifdef __ALINALG_H
#include <alinalg.hpp>
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
	/// Cell centers' coords
	const amat::Matrix<acfd_real>* rc;
	/// Ghost cell centers
	const amat::Matrix<acfd_real>* rcg;
	/// Number of converved variables
	int nvars;
	/// Cell-centered flow vaiables
	const amat::Matrix<acfd_real>* u;
	/// flow variables at ghost cells
	const amat::Matrix<acfd_real>* ug;
	/// Cell-centred x-gradients
	amat::Matrix<acfd_real>* dudx;
	/// Cell-centred y-gradients
	amat::Matrix<acfd_real>* dudy;

public:
	virtual void setup(const UTriMesh* mesh, const amat::Matrix<acfd_real>* unk, const amat::Matrix<acfd_real>* unkg, amat::Matrix<acfd_real>* gradx, amat::Matrix<acfd_real>* grady, 
			const amat::Matrix<acfd_real>* _rc, const amat::Matrix<acfd_real>* const _rcg);
	virtual void compute_gradients() = 0;
};

void Reconstruction::setup(const UTriMesh* mesh, const amat::Matrix<acfd_real>* unk, const amat::Matrix<acfd_real>* unkg, amat::Matrix<acfd_real>* gradx, amat::Matrix<acfd_real>* grady, 
		const amat::Matrix<acfd_real>* _rc, const amat::Matrix<acfd_real>* const _rcg)
{
	m = mesh;
	u = unk;
	ug = unkg;
	dudx = gradx;
	dudy = grady;
	rc = _rc;
	rcg = _rcg;
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

/** The state at the face is approximated as a simple average.
 */
void GreenGaussReconstruction::compute_gradients()
{
	dudx->zeros(); dudy->zeros();
	
	int iface, idim, ielem, jelem, ivar;
	acfd_real areainv1, areainv2;
	std::vector<acfd_real> ut(nvars);
	
	for(iface = 0; iface < m->gnbface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		areainv1 = 1.0/m->gjacobians(ielem);
		
		for(ivar = 0; ivar < nvars; ivar++)
		{
			ut[ivar] = (u->get(ielem,ivar) + ug->get(iface,ivar))*0.5 * m->ggallfa(iface,2);
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
			ut[ivar] = (u->get(ielem,ivar) + u->get(jelem,ivar))*0.5 * m->ggallfa(iface,2);
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
	std::vector<amat::Matrix<acfd_real>> V;		///< LHS of least-squares problem
	std::vector<amat::Matrix<acfd_real>> f;		///< RHS of least-squares problem
	amat::Matrix<acfd_real> d;					///< unknown vector of least-squares problem
	amat::Matrix<acfd_real> idets;				///< inverse of determinants of the LHS
	amat::Matrix<acfd_real> du;

public:
	void setup(UTriMesh* mesh, amat::Matrix<acfd_real>* unk, amat::Matrix<acfd_real>* unkg, amat::Matrix<acfd_real>* gradx, amat::Matrix<acfd_real>* grady, 
			amat::Matrix<acfd_real>* _rc, const amat::Matrix<acfd_real>* const _rcg);
	void compute_gradients();
};

void WeightedLeastSquaresReconstruction::setup(UTriMesh* mesh, amat::Matrix<acfd_real>* unk, amat::Matrix<acfd_real>* unkg, amat::Matrix<acfd_real>* gradx, amat::Matrix<acfd_real>* grady, 
		amat::Matrix<acfd_real>* _rc, const amat::Matrix<acfd_real>* const _rcg)
{
	Reconstruction::setup(mesh, unk, unkg, gradx, grady, _rc, _rcg);

	V.resize(m->gnelem());
	f.resize(m->gnelem());
	for(int i = 0; i < m->gnelem(); i++)
	{
		V[i].setup(2,2);
		f[i].setup(2,nvars);
	}
	d.setup(2,nvars);
	idets.setup(m->gnelem(),1);
	du.setup(nvars,1);

	// compute LHS of least-squares problem
	int iface, ielem, jelem, idim;
	amc_real w2, dr[2];

	for(iface = 0; iface < m->gnbface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		w2 = 0;
		for(idim = 0; idim < 2; idim++)
		{
			w2 += (rc->get(ielem,idim)-rcg->get(iface,idim))*(rc->get(ielem,idim)-rcg->get(iface,idim));
			dr[idim] = rc->get(ielem,idim)-rcg->get(iface,idim);
		}
		w2 = 1.0/w;

		V[ielem](0,0) += w2*dr[0]*dr[0];
		V[ielem](1,1) += w2*dr[1]*dr[1];
		V[ielem](0,1) += w2*dr[0]*dr[1];
		V[ielem](1,0) += w2*dr[0]*dr[1];
	}
	for(iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		jelem = m->gintfac(iface,1);
		w2 = 0;
		for(idim = 0; idim < 2; idim++)
		{
			w2 += (rc->get(ielem,idim)-rc->get(jelem,idim))*(rc->get(ielem,idim)-rc->get(jelem,idim));
			dr[idim] = rc->get(ielem,idim)-rc->get(jelem,idim);
		}
		w2 = 1.0/w;

		V[ielem](0,0) += w2*dr[0]*dr[0];
		V[ielem](1,1) += w2*dr[1]*dr[1];
		V[ielem](0,1) += w2*dr[0]*dr[1];
		V[ielem](1,0) += w2*dr[0]*dr[1];
		
		V[jelem](0,0) += w2*dr[0]*dr[0];
		V[jelem](1,1) += w2*dr[1]*dr[1];
		V[jelem](0,1) += w2*dr[0]*dr[1];
		V[jelem](1,0) += w2*dr[0]*dr[1];
	}

	for(ielem = 0; ielem < m->gnelem(); ielem++)
		idets(ielem) = 1.0/determinant2x2(V[ielem]);
}

void WeightedLeastSquaresReconstruction::compute_gradients()
{
	int iface, ielem, jelem, idim, ivar;
	amc_real w2, dr[2];

	// compute least-squares RHS

	for(iface = 0; iface < m->gnbface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		w2 = 0;
		for(idim = 0; idim < 2; idim++)
		{
			w2 += (rc->get(ielem,idim)-rcg->get(iface,idim))*(rc->get(ielem,idim)-rcg->get(iface,idim));
			dr[idim] = rc->get(ielem,idim)-rcg->get(iface,idim);
			for(ivar = 0; ivar < nvars; ivar++)
				du(ivar) = u->get(ielem,ivar) - ug->get(iface,ivar);
		}
		w2 = 1.0/w;
		
		for(ivar = 0; ivar < nvars; ivar++)
		{
			f[ielem](0,ivar) += w2*dr[0]*du(ivar);
			f[ielem](1,ivar) += w2*dr[1]*du(ivar);
		}
	}
	for(iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		jelem = m->gintfac(iface,1);
		w2 = 0;
		for(idim = 0; idim < 2; idim++)
		{
			w2 += (rc->get(ielem,idim)-rc->get(jelem,idim))*(rc->get(ielem,idim)-rc->get(jelem,idim));
			dr[idim] = rc->get(ielem,idim)-rc->get(jelem,idim);
			for(ivar = 0; ivar < nvars; ivar++)
				du(ivar) = u->get(ielem,ivar) - u->get(jelem,ivar);
		}
		w2 = 1.0/w;

		for(ivar = 0; ivar < nvars; ivar++)
		{
			f[ielem](0,ivar) += w2*dr[0]*du(ivar);
			f[ielem](1,ivar) += w2*dr[1]*du(ivar);
			f[jelem](0,ivar) += w2*dr[0]*du(ivar);
			f[jelem](1,ivar) += w2*dr[1]*du(ivar);
		}
	}

	// solve normal equations by Cramer's rule
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(ivar = 0; ivar < nvars; ivar++)
		{
			d(0,ivar) = (f[ielem].get(0,ivar)*V[ielem].get(1,1) - f[ielem].get(1,ivar)*V[ielem].get(0,1)) * idet.get(ielem);
			d(1,ivar) = (V[ielem].get(0,0)*f[ielem].get(1,ivar) - V[ielem].get(1,0)*f[ielem].get(0,ivar)) * idet.get(ielem);
			(*dudx)(ielem,ivar) = d(0,ivar);
			(*dudy)(ielem,ivar) = d(1,ivar);
		}
	}
}

} // end namespace
