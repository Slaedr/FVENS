#include "alimiter.hpp"

namespace acfd {

FaceDataComputation::FaceDataComputation()
{ }

FaceDataComputation::FaceDataComputation (const UMesh2dh* mesh, const Eigen::Matrix* unknowns, const amat::Matrix<a_real>* unknow_ghost, 
		const amat::Matrix<a_real>* x_deriv, const amat::Matrix<a_real>* y_deriv, 
		const amat::Matrix<a_real>* ghost_centres, const amat::Matrix<a_real>* c_centres, 
		const amat::Matrix<a_real>* gauss_r, amat::Matrix<a_real>* uface_left, amat::Matrix<a_real>* uface_right)
	:
	m(mesh),
	u (unknowns),
	ug (unknow_ghost),          // contains ghost cell states according to BCs for each boundary edge
	dudx (x_deriv),
	dudy (y_deriv),
	rb (ghost_centres),       // contains coords of right "cell centroid" of each boundary edge
	ri (c_centres),
	gr (gauss_r),
	ufl (uface_left),
	ufr (uface_right),
	ng (gauss_r[0].rows())
{ }

FaceDataComputation::~FaceDataComputation()
{ }

void FaceDataComputation::setup(const UMesh2dh* mesh, const Eigen::Matrix* unknowns, const amat::Matrix<a_real>* unknow_ghost, 
		const amat::Matrix<a_real>* x_deriv, const amat::Matrix<a_real>* y_deriv, 
		const amat::Matrix<a_real>* ghost_centres, const amat::Matrix<a_real>* c_centres,
		const amat::Matrix<a_real>* gauss_r, amat::Matrix<a_real>* uface_left, amat::Matrix<a_real>* uface_right)
{
	m = mesh;
	u = unknowns;
	ug = unknow_ghost;          // contains ghost cell states according to BCs for each boundary edge
	dudx = x_deriv;
	dudy = y_deriv;
	ri = c_centres;
	rb = ghost_centres;       // contains x-coord of right "cell centroid" of each boundary edge
	gr = gauss_r;
	ufl = uface_left;
	ufr = uface_right;
	ng = gr[0].rows();
}

NoLimiter::NoLimiter(const UMesh2dh* mesh, const Eigen::Matrix* unknowns, const amat::Matrix<a_real>* unknow_ghost, 
		const amat::Matrix<a_real>* x_deriv, const amat::Matrix<a_real>* y_deriv, 
		const amat::Matrix<a_real>* ghost_centres, const amat::Matrix<a_real>* c_centres, 
		const amat::Matrix<a_real>* gauss_r, amat::Matrix<a_real>* uface_left, amat::Matrix<a_real>* uface_right)
	: FaceDataComputation(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, ghost_centres, c_centres, gauss_r, uface_left, uface_right)
{ }

void NoLimiter::compute_face_values()
{
	// (a) internal faces
	//cout << "NoLimiter: compute_face_values(): Computing values at faces - internal\n";
#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			a_int jelem = m->gintfac(ied,1);

			//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
			for(int ig = 0; ig < NGAUSS; ig++)      // iterate over gauss points
			{
				for(int i = 0; i < NVARS; i++)
				{

					(*ufl)(ied,i) = (*u)(ielem,i) + dudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + dudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
					(*ufr)(ied,i) = (*u)(jelem,i) + dudx->get(jelem,i)*(gr[ied].get(ig,0)-ri->get(jelem,0)) + dudy->get(jelem,i)*(gr[ied].get(ig,1)-ri->get(jelem,1));
				}
			}
		}
		//cout << "NoLimiter: compute_unlimited_interface_values(): Computing values at faces - boundary\n";
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
#pragma omp for
		for(a_int ied = 0; ied < m->gnbface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);

			for(int ig = 0; ig < NGAUSS; ig++)
			{
				for(int i = 0; i < NVARS; i++)
					(*ufl)(ied,i) = (*u)(ielem,i) + dudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + dudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
			}
		}
	}
}

WENOLimiter::WENOLimiter(const UMesh2dh* mesh, const Eigen::Matrix* unknowns, const amat::Matrix<a_real>* unknow_ghost, 
		const amat::Matrix<a_real>* x_deriv, const amat::Matrix<a_real>* y_deriv, 
		const amat::Matrix<a_real>* ghost_centres, const amat::Matrix<a_real>* c_centres, const amat::Matrix<a_real>* gauss_r, 
		amat::Matrix<a_real>* uface_left, amat::Matrix<a_real>* uface_right)
	: FaceDataComputation(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, ghost_centres, c_centres, gauss_r, uface_left, uface_right)
{
	ldudx = new amat::Matrix<a_real>(m->gnelem(),NVARS);
	ldudy = new amat::Matrix<a_real>(m->gnelem(),NVARS);
	// values below chosen from second reference (Dumbser and Kaeser)
	gamma = 4.0;
	lambda = 1e3;
	epsilon = 1e-5;
}

WENOLimiter::~WENOLimiter()
{
	delete ldudx;
	delete ldudy;
}

void WENOLimiter::compute_face_values()
{
	// first compute limited derivatives at each cell

#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
		{
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				a_real wsum = 0;
				(*ldudx)(ielem,ivar) = 0;
				(*ldudy)(ielem,ivar) = 0;

				// Central stencil
				a_real denom = pow( dudx->get(ielem,ivar)*dudx->get(ielem,ivar) + dudy->get(ielem,ivar)*dudy->get(ielem,ivar) + epsilon, gamma);
				a_real w = lambda / denom;
				wsum += w;
				(*ldudx)(ielem,ivar) += w*dudx->get(ielem,ivar);
				(*ldudy)(ielem,ivar) += w*dudy->get(ielem,ivar);

				// Biased stencils
				for(int jel = 0; jel < m->gnfael(ielem); jel++)
				{
					a_int jelem = m->gesuel(ielem,jel);

					// ignore ghost cells
					if(jelem >= m->gnelem())
						continue;

					denom = pow( dudx->get(jelem,ivar)*dudx->get(jelem,ivar) + dudy->get(jelem,ivar)*dudy->get(jelem,ivar) + epsilon, gamma);
					w = 1.0 / denom;
					wsum += w;
					(*ldudx)(ielem,ivar) += w*dudx->get(jelem,ivar);
					(*ldudy)(ielem,ivar) += w*dudy->get(jelem,ivar);
				}

				(*ldudx)(ielem,ivar) /= wsum;
				(*ldudy)(ielem,ivar) /= wsum;
			}
		}
		
		// internal faces
#pragma omp for
		for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			a_int jelem = m->gintfac(ied,1);

			//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
			for(int ig = 0; ig < NGAUSS; ig++)      // iterate over gauss points
			{
				for(int i = 0; i < NVARS; i++)
				{

					(*ufl)(ied,i) = (*u)(ielem,i) + ldudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + ldudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
					(*ufr)(ied,i) = (*u)(jelem,i) + ldudx->get(jelem,i)*(gr[ied].get(ig,0)-ri->get(jelem,0)) + ldudy->get(jelem,i)*(gr[ied].get(ig,1)-ri->get(jelem,1));
				}
			}
		}
		
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
#pragma omp for
		for(a_int ied = 0; ied < m->gnbface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);

			for(int ig = 0; ig < NGAUSS; ig++)
			{
				for(int i = 0; i < NVARS; i++)
					(*ufl)(ied,i) = (*u)(ielem,i) + ldudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + ldudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
			}
		}
	} // end parallel region
}

void VanAlbadaLimiter::setup(const UMesh2dh* mesh, const Eigen::Matrix* unknowns, const amat::Matrix<a_real>* unknow_ghost, 
	const amat::Matrix<a_real>* x_deriv, const amat::Matrix<a_real>* y_deriv, 
	const amat::Matrix<a_real>* ghost_centres, const amat::Matrix<a_real>* r_centres, 
	const amat::Matrix<a_real>* gauss_r, amat::Matrix<a_real>* uface_left, amat::Matrix<a_real>* uface_right)
{
	FaceDataComputation::setup(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, ghost_centres, r_centres, gauss_r, uface_left, uface_right);
	eps = 1e-8;
	k = 1.0/3.0;
	phi_l.setup(m->gnaface(), NVARS);
	phi_r.setup(m->gnaface(), NVARS);
}

/// Calculate values of variables at left and right sides of each face based on computed derivatives and limiter values
void VanAlbadaLimiter::compute_face_values()
{
	//compute_limiters
	
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		int lel = m->gintfac(ied,0);
		amat::Matrix<a_real> deltam(NVARS,1);
		for(int i = 0; i < NVARS; i++)
		{
			deltam(i) = 2 * ( dudx->get(lel,i)*(rb->get(ied,0)-ri->get(lel,0)) + dudy->get(lel,i)*(rb->get(ied,1)-ri->get(lel,1)) ) - (ug->get(ied,i) - (*u)(lel,i));
			phi_l(ied,i) = (2*deltam(i) * (ug->get(ied,i) - (*u)(lel,i)) + eps) / (deltam(i)*deltam(i) + (ug->get(ied,i) - (*u)(lel,i))*(ug->get(ied,i) - (*u)(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;
		}
	}

	for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		a_int lel = m->gintfac(ied,0);
		a_int rel = m->gintfac(ied,1);
		amat::Matrix<a_real> deltam(NVARS,1);
		amat::Matrix<a_real> deltap(NVARS,1);
		for(int i = 0; i < NVARS; i++)
		{
			deltam(i) = 2 * ( dudx->get(lel,i)*(ri->get(rel,0)-ri->get(lel,0)) + dudy->get(lel,i)*(ri->get(rel,1)-ri->get(lel,1)) ) - ((*u)(rel,i) - (*u)(lel,i));
			deltap(i) = 2 * ( dudx->get(rel,i)*(ri->get(rel,0)-ri->get(lel,0)) + dudy->get(rel,i)*(ri->get(rel,1)-ri->get(lel,1)) ) - ((*u)(rel,i) - (*u)(lel,i));

			phi_l(ied,i) = (2*deltam(i) * ((*u)(rel,i) - (*u)(lel,i)) + eps) / (deltam(i)*deltam(i) + ((*u)(rel,i) - (*u)(lel,i))*((*u)(rel,i) - (*u)(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;

			phi_r(ied,i) = (2*deltap(i) * ((*u)(rel,i) - (*u)(lel,i)) + eps) / (deltap(i)*deltap(i) + ((*u)(rel,i) - (*u)(lel,i))*((*u)(rel,i) - (*u)(lel,i)) + eps);
			if( phi_r(ied,i) < 0.0) phi_r(ied,i) = 0.0;
		}
	}

	// apply the limiters
	
	//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - internal\n";
	amat::Matrix<a_real> deltam(NVARS,1);
	amat::Matrix<a_real> deltap(NVARS,1);
	for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		a_int ielem = m->gintfac(ied,0);
		a_int jelem = m->gintfac(ied,1);

		// NOTE: Only for 1 Gauss point per face
		//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
		for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
		{
			for(int i = 0; i < NVARS; i++)
			{
				deltam(i) = 2 * ( dudx->get(ielem,i)*(ri->get(jelem,0)-ri->get(ielem,0)) + dudy->get(ielem,i)*(ri->get(jelem,1)-ri->get(ielem,1)) ) - ((*u)(jelem,i) - (*u)(ielem,i));
				deltap(i) = 2 * ( dudx->get(jelem,i)*(ri->get(jelem,0)-ri->get(ielem,0)) + dudy->get(jelem,i)*(ri->get(jelem,1)-ri->get(ielem,1)) ) - ((*u)(jelem,i) - (*u)(ielem,i));

				(*ufl)(ied,i) = (*u)(ielem,i) + phi_l.get(ied,i)/4.0*( (1-k*phi_l.get(ied,i))*deltam.get(i) + (1+k*phi_l.get(ied,i))*((*u)(jelem,i) - (*u)(ielem,i)) );
				(*ufr)(ied,i) = (*u)(jelem,i) + phi_r.get(ied,i)/4.0*( (1-k*phi_r.get(ied,i))*deltap(i) + (1+k*phi_r.get(ied,i))*((*u)(jelem,i) - (*u)(ielem,i)) );
			}
		}
	}
}

} // end namespace
