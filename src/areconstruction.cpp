#include "areconstruction.hpp"

namespace acfd {

SolutionReconstruction::SolutionReconstruction (const UMesh2dh* mesh, 
		const amat::Array2d<a_real>* c_centres, 
		const amat::Array2d<a_real>* gauss_r)
	: m{mesh}, ri{c_centres}, gr{gauss_r}, ng{gr[0].rows()}
{ }

SolutionReconstruction::~SolutionReconstruction()
{ }

LinearUnlimitedReconstruction::LinearUnlimitedReconstruction(const UMesh2dh* mesh,
		const amat::Array2d<a_real>* c_centres, const amat::Array2d<a_real>* gauss_r)
	: SolutionReconstruction(mesh, c_centres, gauss_r)
{ }

void LinearUnlimitedReconstruction::compute_face_values(
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug,
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy, 
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
	// (a) internal faces
#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			a_int jelem = m->gintfac(ied,1);

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

WENOReconstruction::WENOReconstruction(const UMesh2dh* mesh,
		const amat::Array2d<a_real>* c_centres, const amat::Array2d<a_real>* gauss_r)
	: SolutionReconstruction(mesh, c_centres, gauss_r),
	  gamma{4.0}, lambda{1.0e3}, epsilon{1.0e-5}
{
}

void WENOReconstruction::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug,
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy, 
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
	// first compute limited derivatives at each cell

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			a_real wsum = 0;
			a_real ldudx = 0;
			a_real ldudy = 0;

			// Central stencil
			a_real denom = pow( dudx(ielem,ivar)*dudx(ielem,ivar) 
					+ dudy(ielem,ivar)*dudy(ielem,ivar) + epsilon, gamma);
			a_real w = lambda / denom;
			wsum += w;
			ldudx += w*dudx(ielem,ivar);
			ldudy += w*dudy(ielem,ivar);

			// Biased stencils
			for(int jel = 0; jel < m->gnfael(ielem); jel++)
			{
				const a_int jelem = m->gesuel(ielem,jel);

				// ignore ghost cells
				if(jelem >= m->gnelem())
					continue;

				denom = pow( dudx(jelem,ivar)*dudx(jelem,ivar) 
						+ dudy(jelem,ivar)*dudy(jelem,ivar) + epsilon, gamma);
				w = 1.0 / denom;
				wsum += w;
				ldudx += w*dudx(jelem,ivar);
				ldudy += w*dudy(jelem,ivar);
			}

			ldudx /= wsum;
			ldudy /= wsum;
			
			for(int j = 0; j < m->gnfael(ielem); j++)
			{
				const a_int face = m->gelemface(ielem,j);
				const a_int jelem = m->gesuel(ielem,j);
				
				if(ielem < jelem)
					ufl(face,ivar) = u(ielem,ivar) 
						+ ldudx*(gr[face](0,0)-(*ri)(ielem,0))
						+ ldudy*(gr[face](0,1)-(*ri)(ielem,1));
				else
					ufr(face,ivar) = u(ielem,ivar) 
						+ ldudx*(gr[face](0,0)-(*ri)(ielem,0))
						+ ldudy*(gr[face](0,1)-(*ri)(ielem,1));
			}
		}
	}
}

MUSCLReconstruction::MUSCLReconstruction(const UMesh2dh* mesh,
		const amat::Array2d<a_real>* r_centres, const amat::Array2d<a_real>* gauss_r)
	: SolutionReconstruction(mesh, r_centres, gauss_r), eps{1e-8}, k{1.0/3.0}
{ }

inline
a_real MUSCLReconstruction::computeBiasedDifference(const a_real *const ri, const a_real *const rj,
		const a_real ui, const a_real uj, const a_real *const grads) const
{
	a_real del = 0;
	for(int idim = 0; idim < NDIM; idim++)
		del += grads[idim]*(rj[idim]-ri[idim]);

	return 2.0*del - (uj-ui);
}

inline
a_real MUSCLReconstruction::musclReconstructLeft(const a_real ui, const a_real uj, 
			const a_real deltam, const a_real phi) const
{
	return ui + phi/4.0*( (1.0-k*phi)*deltam + (1.0+k*phi)*(uj - ui) );
}

inline
a_real MUSCLReconstruction::musclReconstructRight(const a_real ui, const a_real uj, 
			const a_real deltap, const a_real phi) const
{
	return uj - phi/4.0*( (1.0-k*phi)*deltap + (1.0+k*phi)*(uj - ui) );
}

MUSCLVanAlbada::MUSCLVanAlbada(const UMesh2dh* mesh,
		const amat::Array2d<a_real>* r_centres, const amat::Array2d<a_real>* gauss_r)
	: MUSCLReconstruction(mesh, r_centres, gauss_r)
{ }

void MUSCLVanAlbada::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug,
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy, 
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_int jelem = m->gintfac(ied,1);

		for(int i = 0; i < NVARS; i++)
		{
			const a_real grads[NDIM] = {dudx(ielem,i), dudy(ielem,i)};
			
			const a_real deltam = computeBiasedDifference(&(*ri)(ielem,0), &(*ri)(jelem,0),
					u(ielem,i), ug(ied,i), grads);
			
			a_real phi_l = (2*deltam * (ug(ied,i) - u(ielem,i)) + eps) 
				/ (deltam*deltam + (ug(ied,i) - u(ielem,i))*(ug(ied,i) - u(ielem,i)) + eps);
			if( phi_l < 0.0) phi_l = 0.0;

			ufl(ied,i) = musclReconstructLeft(u(ielem,i), ug(ied,i), deltam, phi_l);
		}
	}
	
#pragma omp parallel for default(shared)
	for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_int jelem = m->gintfac(ied,1);

		for(int i = 0; i < NVARS; i++)
		{
			const a_real gradsl[NDIM] = {dudx(ielem,i), dudy(ielem,i)};
			const a_real gradsr[NDIM] = {dudx(jelem,i), dudy(jelem,i)};

			const a_real deltam = computeBiasedDifference(&(*ri)(ielem,0), &(*ri)(jelem,0),
					u(ielem,i), u(jelem,i), gradsl);
			const a_real deltap = computeBiasedDifference(&(*ri)(ielem,0), &(*ri)(jelem,0),
					u(ielem,i), u(jelem,i), gradsr);
			
			a_real phi_l = (2*deltam * (u(jelem,i) - u(ielem,i)) + eps) 
				/ (deltam*deltam + (u(jelem,i) - u(ielem,i))*(u(jelem,i) - u(ielem,i)) + eps);
			if( phi_l < 0.0) phi_l = 0.0;

			a_real phi_r = (2*deltap * (u(jelem,i) - u(ielem,i)) + eps) 
				/ (deltap*deltap + (u(jelem,i) - u(ielem,i))*(u(jelem,i) - u(ielem,i)) + eps);
			if( phi_r < 0.0) phi_r = 0.0;

			ufl(ied,i) = musclReconstructLeft(u(ielem,i), u(jelem,i), deltam, phi_l);
			ufr(ied,i) = musclReconstructRight(u(ielem,i), u(jelem,i), deltap, phi_r);
		}
	}
}

BarthJespersenLimiter::BarthJespersenLimiter(const UMesh2dh* mesh, 
		const amat::Array2d<a_real>* r_centres, const amat::Array2d<a_real>* gauss_r)
	: SolutionReconstruction(mesh, r_centres, gauss_r)
{
}

void BarthJespersenLimiter::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug, 
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy,
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			a_real duimin=0, duimax=0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int jel = m->gesuel(iel,j);
				const a_real dui = u(jel,ivar)-u(iel,ivar);
				if(dui > duimax) duimax = dui;
				if(dui < duimin) duimin = dui;
			}
			
			a_real lim = 1.0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int face = m->gelemface(iel,j);
				const a_real uface = u(iel,ivar) + dudx(iel,ivar)*(gr[face](0,0)-(*ri)(iel,0))
					+ dudy(iel,ivar)*(gr[face](0,1)-(*ri)(iel,1));
				
				a_real phiik;
				const a_real diff = uface - u(iel,ivar);
				if(diff>0)
					phiik = 1 < duimax/diff ? 1 : duimax/diff;
				else if(diff < 0)
					phiik = 1 < duimin/diff ? 1 : duimin/diff;
				else
					phiik = 1;

				if(phiik < lim)
					lim = phiik;
			}
			
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int face = m->gelemface(iel,j);
				const a_int jel = m->gesuel(iel,j);
				
				if(iel < jel)
					ufl(face,ivar) = u(iel,ivar) 
						+ lim*dudx(iel,ivar)*(gr[face](0,0)-(*ri)(iel,0))
						+ lim*dudy(iel,ivar)*(gr[face](0,1)-(*ri)(iel,1));
				else
					ufr(face,ivar) = u(iel,ivar) 
						+ lim*dudx(iel,ivar)*(gr[face](0,0)-(*ri)(iel,0))
						+ lim*dudy(iel,ivar)*(gr[face](0,1)-(*ri)(iel,1));
			}

		}
	}
}

VenkatakrishnanLimiter::VenkatakrishnanLimiter(const UMesh2dh* mesh, 
		const amat::Array2d<a_real>* r_centres, const amat::Array2d<a_real>* gauss_r,
		a_real k_param=2.0)
	: SolutionReconstruction(mesh, r_centres, gauss_r), K{k_param}
{
	std::cout << "  Venkatakrishnan Limiter: Constant K = " << K << std::endl;
	// compute characteristic length, currently the maximum edge length, of all cells
	clength.resize(m->gnelem());
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int ifa = 0; ifa < m->gnnode(iel); ifa++)
		{
			a_real llen = 0;
			const int inode = ifa, jnode = (ifa+1) % m->gnnode(iel);
			for(int idim = 0; idim < 2; idim++)
				llen += std::pow(m->gcoords(m->ginpoel(iel,inode),idim) 
						- m->gcoords(m->ginpoel(iel,jnode),idim), 2);

			if(clength[iel] < llen) clength[iel] = llen;
		}
		clength[iel] = std::sqrt(clength[iel]);
	}
}

void VenkatakrishnanLimiter::compute_face_values(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
		const amat::Array2d<a_real>& ug, 
		const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy,
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		const a_real eps2 = std::pow(K*clength[iel], 3);

		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			a_real duimin=0, duimax=0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int jel = m->gesuel(iel,j);
				const a_real dui = u(jel,ivar)-u(iel,ivar);
				if(dui > duimax) duimax = dui;
				if(dui < duimin) duimin = dui;
			}
			
			a_real lim = 1.0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int face = m->gelemface(iel,j);
				const a_real uface = u(iel,ivar) + dudx(iel,ivar)*(gr[face](0,0)-(*ri)(iel,0))
					+ dudy(iel,ivar)*(gr[face](0,1)-(*ri)(iel,1));
				
				const a_real dm = uface - u(iel,ivar);

				// Venkatakrishnan modification
				const a_real dp = dm < 0 ? duimin : duimax;
				const a_real phiik = (dp*dp + 2*dp*dm + eps2)/(dp*dp + dp*dm + 2*dm*dm + eps2);

				if(phiik < lim)
					lim = phiik;
			}
			
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int face = m->gelemface(iel,j);
				const a_int jel = m->gesuel(iel,j);
				
				if(iel < jel)
					ufl(face,ivar) = u(iel,ivar) 
						+ lim*dudx(iel,ivar)*(gr[face](0,0)-(*ri)(iel,0))
						+ lim*dudy(iel,ivar)*(gr[face](0,1)-(*ri)(iel,1));
				else
					ufr(face,ivar) = u(iel,ivar) 
						+ lim*dudx(iel,ivar)*(gr[face](0,0)-(*ri)(iel,0))
						+ lim*dudy(iel,ivar)*(gr[face](0,1)-(*ri)(iel,1));
			}

		}
	}
}

} // end namespace

