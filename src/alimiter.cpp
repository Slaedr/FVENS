#include "alimiter.hpp"

namespace acfd {

FaceDataComputation::FaceDataComputation()
{ }

FaceDataComputation::FaceDataComputation (const UMesh2dh* mesh, 
		const amat::Array2d<a_real>* ghost_centres, 
		const amat::Array2d<a_real>* c_centres, 
		const amat::Array2d<a_real>* gauss_r)
	:
	m(mesh),
	rb (ghost_centres),       // contains coords of right "cell centroid" of each boundary edge
	ri (c_centres),
	gr (gauss_r),
	ng (gauss_r[0].rows())
{ }

FaceDataComputation::~FaceDataComputation()
{ }

void FaceDataComputation::setup(const UMesh2dh* mesh,
		const amat::Array2d<a_real>* ghost_centres, const amat::Array2d<a_real>* c_centres,
		const amat::Array2d<a_real>* gauss_r)
{
	m = mesh;
	ri = c_centres;
	rb = ghost_centres;       // contains x-coord of right "cell centroid" of each boundary edge
	gr = gauss_r;
	ng = gr[0].rows();
}

NoLimiter::NoLimiter(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, 
		const amat::Array2d<a_real>* c_centres, const amat::Array2d<a_real>* gauss_r)
	: FaceDataComputation(mesh, ghost_centres, c_centres, gauss_r)
{ }

void NoLimiter::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug,
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy, 
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr)
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

					ufl(ied,i) = u(ielem,i) 
						+ dudx(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) 
						+ dudy(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
					ufr(ied,i) = u(jelem,i) 
						+ dudx(jelem,i)*(gr[ied].get(ig,0)-ri->get(jelem,0)) 
						+ dudy(jelem,i)*(gr[ied].get(ig,1)-ri->get(jelem,1));
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
					ufl(ied,i) = u(ielem,i) 
						+ dudx(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) 
						+ dudy(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
			}
		}
	}
}

WENOLimiter::WENOLimiter(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, 
		const amat::Array2d<a_real>* c_centres, const amat::Array2d<a_real>* gauss_r)
	: FaceDataComputation(mesh, ghost_centres, c_centres, gauss_r)
{
	ldudx.resize(m->gnelem(),NVARS);
	ldudy.resize(m->gnelem(),NVARS);
	// values below chosen from second reference (Dumbser and Kaeser)
	gamma = 4.0;
	lambda = 1e3;
	epsilon = 1e-5;
}

void WENOLimiter::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug,
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy, 
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr)
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
				ldudx(ielem,ivar) = 0;
				ldudy(ielem,ivar) = 0;

				// Central stencil
				a_real denom = pow( dudx(ielem,ivar)*dudx(ielem,ivar) 
						+ dudy(ielem,ivar)*dudy(ielem,ivar) + epsilon, gamma);
				a_real w = lambda / denom;
				wsum += w;
				ldudx(ielem,ivar) += w*dudx(ielem,ivar);
				ldudy(ielem,ivar) += w*dudy(ielem,ivar);

				// Biased stencils
				for(int jel = 0; jel < m->gnfael(ielem); jel++)
				{
					a_int jelem = m->gesuel(ielem,jel);

					// ignore ghost cells
					if(jelem >= m->gnelem())
						continue;

					denom = pow( dudx(jelem,ivar)*dudx(jelem,ivar) 
							+ dudy(jelem,ivar)*dudy(jelem,ivar) + epsilon, gamma);
					w = 1.0 / denom;
					wsum += w;
					ldudx(ielem,ivar) += w*dudx(jelem,ivar);
					ldudy(ielem,ivar) += w*dudy(jelem,ivar);
				}

				ldudx(ielem,ivar) /= wsum;
				ldudy(ielem,ivar) /= wsum;
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

					ufl(ied,i) = u(ielem,i) 
						+ ldudx(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) 
						+ ldudy(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
					ufr(ied,i) = u(jelem,i) 
						+ ldudx(jelem,i)*(gr[ied].get(ig,0)-ri->get(jelem,0)) 
						+ ldudy(jelem,i)*(gr[ied].get(ig,1)-ri->get(jelem,1));
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
					ufl(ied,i) = u(ielem,i) 
						+ ldudx(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) 
						+ ldudy(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
			}
		}
	} // end parallel region
}

VanAlbadaLimiter::VanAlbadaLimiter(const UMesh2dh* mesh, const amat::Array2d<a_real>* ghost_centres, 
		const amat::Array2d<a_real>* r_centres, const amat::Array2d<a_real>* gauss_r)
	: FaceDataComputation(mesh, ghost_centres, r_centres, gauss_r)
{
	eps = 1e-8;
	k = 1.0/3.0;
	phi_l.resize(m->gnaface(), NVARS);
	phi_r.resize(m->gnaface(), NVARS);
}

void VanAlbadaLimiter::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug,
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy, 
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr)
{
	//compute_limiters
	
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		int lel = m->gintfac(ied,0);
		for(int i = 0; i < NVARS; i++)
		{
			a_real deltam;
			deltam = 2 * ( dudx(lel,i)*(rb->get(ied,0)-ri->get(lel,0)) 
				+ dudy(lel,i)*(rb->get(ied,1)-ri->get(lel,1)) ) 
				- (ug(ied,i) - u(lel,i));
			phi_l(ied,i) = (2*deltam * (ug(ied,i) - u(lel,i)) + eps) 
				/ (deltam*deltam + (ug(ied,i) - u(lel,i))*(ug(ied,i) - u(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;
		}
	}

	for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		a_int lel = m->gintfac(ied,0);
		a_int rel = m->gintfac(ied,1);
		for(int i = 0; i < NVARS; i++)
		{
			a_real deltam, deltap;
			deltam = 2 * ( dudx(lel,i)*(ri->get(rel,0)-ri->get(lel,0)) 
					+ dudy(lel,i)*(ri->get(rel,1)-ri->get(lel,1)) ) - (u(rel,i) - u(lel,i));
			deltap = 2 * ( dudx(rel,i)*(ri->get(rel,0)-ri->get(lel,0)) 
					+ dudy(rel,i)*(ri->get(rel,1)-ri->get(lel,1)) ) - (u(rel,i) - u(lel,i));

			phi_l(ied,i) = (2*deltam * (u(rel,i) - u(lel,i)) + eps) 
				/ (deltam*deltam + (u(rel,i) - u(lel,i))*(u(rel,i) - u(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;

			phi_r(ied,i) = (2*deltap * (u(rel,i) - u(lel,i)) + eps) 
				/ (deltap*deltap + (u(rel,i) - u(lel,i))*(u(rel,i) - u(lel,i)) + eps);
			if( phi_r(ied,i) < 0.0) phi_r(ied,i) = 0.0;
		}
	}

	// apply the limiters
	
	//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - internal\n";
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
				a_real deltam, deltap;
				deltam = 2 * ( dudx(ielem,i)*(ri->get(jelem,0)-ri->get(ielem,0)) 
					+ dudy(ielem,i)*(ri->get(jelem,1)-ri->get(ielem,1)) ) 
					- (u(jelem,i) - u(ielem,i));
				deltap = 2 * ( dudx(jelem,i)*(ri->get(jelem,0)-ri->get(ielem,0)) 
					+ dudy(jelem,i)*(ri->get(jelem,1)-ri->get(ielem,1)) ) 
					- (u(jelem,i) - u(ielem,i));

				ufl(ied,i) = u(ielem,i) + phi_l(ied,i)/4.0
					*( (1-k*phi_l(ied,i))*deltam + (1+k*phi_l(ied,i))*(u(jelem,i) - u(ielem,i)) );
				ufr(ied,i) = u(jelem,i) + phi_r(ied,i)/4.0
					*( (1-k*phi_r(ied,i))*deltap + (1+k*phi_r(ied,i))*(u(jelem,i) - u(ielem,i)) );
			}
		}
	}
}

} // end namespace
