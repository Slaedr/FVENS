/** \file areconstruction.cpp
 * \brief Implementation of solution reconstruction schemes (limiters)
 * \author Aditya Kashi
 */

#include <iostream>
#include "areconstruction.hpp"

namespace acfd {

/// Reconstructs a face value
static inline a_real linearExtrapolate(
		const a_real ucell,             ///< Relevant cell centred value
		const FArray<NDIM,NVARS>& grad, ///< Gradients
		const int ivar,                 ///< Index of physical variable to be reconstructed
		const a_real lim,               ///< Limiter value
		const a_real *const gp,         ///< Quadrature point coords
		const a_real *const rc          ///< Cell centre coords
	)
{
	a_real uface = ucell;
	for(int idim = 0; idim < NDIM; idim++)
		uface += lim*grad(idim,ivar)*(gp[idim] - rc[idim]);
	return uface;
}

SolutionReconstruction::SolutionReconstruction (const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& c_centres, 
		const amat::Array2d<a_real>* gauss_r)
	: m{mesh}, ri{c_centres}, gr{gauss_r}, ng{gr[0].rows()}
{ }

SolutionReconstruction::~SolutionReconstruction()
{ }

LinearUnlimitedReconstruction::LinearUnlimitedReconstruction(const UMesh2dh *const mesh,
		const amat::Array2d<a_real>& c_centres, const amat::Array2d<a_real>* gauss_r)
	: SolutionReconstruction(mesh, c_centres, gauss_r)
{ }

void LinearUnlimitedReconstruction::compute_face_values(
		const MVector& u, 
		const amat::Array2d<a_real>& ug,
		const std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>>& grads,
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

			for(int i = 0; i < NVARS; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
						&gr[ied](0,0), &ri(ielem,0));
				ufr(ied,i) = linearExtrapolate(u(jelem,i), grads[jelem], i, 1.0,
						&gr[ied](0,0), &ri(jelem,0));
			}
		}
		
#pragma omp for
		for(a_int ied = 0; ied < m->gnbface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);

			for(int i = 0; i < NVARS; i++) 
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
						&gr[ied](0,0), &ri(ielem,0));
			}
		}
	}
}

WENOReconstruction::WENOReconstruction(const UMesh2dh *const mesh,
		const amat::Array2d<a_real>& c_centres, const amat::Array2d<a_real>* gauss_r, const a_real l)
	: SolutionReconstruction(mesh, c_centres, gauss_r),
	  gamma{4.0}, lambda{l}, epsilon{1.0e-5}
{
}

/// Returns the squared magnitude of the gradient of a variable
/** \param[in] grad The gradient array
 * \param[in] ivar The index of the physical variable whose gradient magnitude is needed
 */
static inline a_real gradientMagnitude2(const FArray<NDIM,NVARS>& grad, const int ivar) {
	a_real res = 0;
	for(int j = 0; j < NDIM; j++)
		res += grad(j,ivar)*grad(j,ivar);
	return res;
}

void WENOReconstruction::compute_face_values(const MVector& u, 
		const amat::Array2d<a_real>& ug,
		const std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>>& grads,
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
	// first compute limited derivatives at each cell

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			a_real wsum = 0;
			a_real lgrad[NDIM]; 
			zeros(lgrad, NDIM);

			// Central stencil
			const a_real denom = pow( gradientMagnitude2(grads[ielem],ivar) + epsilon , gamma );
			const a_real w = lambda / denom;
			wsum += w;
			for(int j = 0; j < NDIM; j++)
				lgrad[j] += w*grads[ielem](j,ivar);

			// Biased stencils
			for(int jel = 0; jel < m->gnfael(ielem); jel++)
			{
				const a_int jelem = m->gesuel(ielem,jel);

				// ignore ghost cells
				if(jelem >= m->gnelem())
					continue;

				const a_real denom = pow( gradientMagnitude2(grads[jelem],ivar) + epsilon , gamma );
				const a_real w = 1.0 / denom;
				wsum += w;
				for(int j = 0; j < NDIM; j++)
					lgrad[j] += w*grads[jelem](j,ivar);
			}

			for(int j = 0; j < NDIM; j++)
				lgrad[j] /= wsum;
			
			for(int j = 0; j < m->gnfael(ielem); j++)
			{
				const a_int face = m->gelemface(ielem,j);
				const a_int jelem = m->gesuel(ielem,j);
				
				if(ielem < jelem) {
					ufl(face,ivar) = u(ielem,ivar);
					for(int j = 0; j < NDIM; j++)
						ufl(face,ivar) += lgrad[j]*(gr[face](0,j) - ri(ielem,j));
				}
				else {
					ufr(face,ivar) = u(ielem,ivar);
					for(int j = 0; j < NDIM; j++)
						ufr(face,ivar) += lgrad[j]*(gr[face](0,j)-ri(ielem,j));
				}
			}
		}
	}
}

MUSCLReconstruction::MUSCLReconstruction(const UMesh2dh *const mesh,
		const amat::Array2d<a_real>& r_centres, const amat::Array2d<a_real>* gauss_r)
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

MUSCLVanAlbada::MUSCLVanAlbada(const UMesh2dh *const mesh,
		const amat::Array2d<a_real>& r_centres, const amat::Array2d<a_real>* gauss_r)
	: MUSCLReconstruction(mesh, r_centres, gauss_r)
{ }

void MUSCLVanAlbada::compute_face_values(const MVector& u, 
		const amat::Array2d<a_real>& ug,
		const std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>>& grads,
		amat::Array2d<a_real>& ufl, amat::Array2d<a_real>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_int jelem = m->gintfac(ied,1);

		for(int i = 0; i < NVARS; i++)
		{
			// Note that the copy below is necessary because grads[ielem] is column major
			a_real grad[NDIM];
			for(int j = 0; j < NDIM; j++)
				grad[j] = grads[ielem](j,i);
			
			const a_real deltam = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), ug(ied,i), grad);
			
			a_real phi_l = (2.0*deltam * (ug(ied,i) - u(ielem,i)) + eps) 
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
			// Note that the copy below is necessary because grads[ielem] is column major
			a_real gradl[NDIM], gradr[NDIM];
			for(int j = 0; j < NDIM; j++) {
				gradl[j] = grads[ielem](j,i);
				gradr[j] = grads[jelem](j,i);
			}

			const a_real deltam = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), u(jelem,i), gradl);
			const a_real deltap = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), u(jelem,i), gradr);
			
			a_real phi_l = (2.0*deltam * (u(jelem,i) - u(ielem,i)) + eps) 
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

BarthJespersenLimiter::BarthJespersenLimiter(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& r_centres, const amat::Array2d<a_real>* gauss_r)
	: SolutionReconstruction(mesh, r_centres, gauss_r)
{
}

void BarthJespersenLimiter::compute_face_values(const MVector& u, 
		const amat::Array2d<a_real>& ug, 
		const std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>>& grads,
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
				
				const a_real uface = linearExtrapolate(u(iel,ivar), grads[iel], ivar, 1.0,
						&gr[face](0,0), &ri(iel,0));
				
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
					ufl(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr[face](0,0), &ri(iel,0));
				else
					ufr(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr[face](0,0), &ri(iel,0));
			}

		}
	}
}

VenkatakrishnanLimiter::VenkatakrishnanLimiter(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& r_centres, const amat::Array2d<a_real>* gauss_r,
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

void VenkatakrishnanLimiter::compute_face_values(const MVector& u, 
		const amat::Array2d<a_real>& ug, 
		const std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>>& grads,
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
				
				const a_real uface = linearExtrapolate(u(iel,ivar), grads[iel], ivar, 1.0,
						&gr[face](0,0), &ri(iel,0));
				
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
					ufl(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr[face](0,0), &ri(iel,0));
				else
					ufr(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr[face](0,0), &ri(iel,0));
			}

		}
	}
}

} // end namespace

