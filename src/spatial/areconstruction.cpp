/** \file areconstruction.cpp
 * \brief Implementation of solution reconstruction schemes (limiters)
 * \author Aditya Kashi
 */

#include <iostream>
#include "mathutils.hpp"
#include "areconstruction.hpp"

namespace fvens {

/// Reconstructs a face value
template <typename scalar>
static inline scalar linearExtrapolate(
		const scalar ucell,                          ///< Relevant cell centred value
		const Eigen::Array<scalar,NDIM,NVARS>& grad, ///< Gradients
		const int ivar,                              ///< Index of physical variable to be reconstructed
		const scalar lim,                            ///< Limiter value
		const scalar *const gp,                      ///< Quadrature point coords
		const scalar *const rc                       ///< Cell centre coords
	)
{
	scalar uface = ucell;
	for(int idim = 0; idim < NDIM; idim++)
		uface += lim*grad(idim,ivar)*(gp[idim] - rc[idim]);
	return uface;
}

template <typename scalar>
SolutionReconstruction<scalar>::SolutionReconstruction (const UMesh2dh<scalar> *const mesh, 
		const amat::Array2d<scalar>& c_centres, 
		const amat::Array2d<scalar>* gauss_r)
	: m{mesh}, ri{c_centres}, gr{gauss_r}, ng{gr[0].rows()}
{ }

template <typename scalar>
SolutionReconstruction<scalar>::~SolutionReconstruction()
{ }

template <typename scalar>
LinearUnlimitedReconstruction<scalar>
::LinearUnlimitedReconstruction(const UMesh2dh<scalar> *const mesh,
                                                             const amat::Array2d<scalar>& c_centres,
                                                             const amat::Array2d<scalar>* gauss_r)
	: SolutionReconstruction<scalar>(mesh, c_centres, gauss_r)
{ }

template <typename scalar>
void LinearUnlimitedReconstruction<scalar>::compute_face_values(
		const MVector<scalar>& u, 
		const amat::Array2d<scalar>& ug,
		const GradArray<scalar,NVARS>& grads,
		amat::Array2d<scalar>& ufl, amat::Array2d<scalar>& ufr) const
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

template <typename scalar>
WENOReconstruction<scalar>::WENOReconstruction(const UMesh2dh<scalar> *const mesh,
                                       const amat::Array2d<scalar>& c_centres,
                                       const amat::Array2d<scalar>* gauss_r, const a_real l)
	: SolutionReconstruction<scalar>(mesh, c_centres, gauss_r),
	  gamma{4.0}, lambda{l}, epsilon{1.0e-5}
{
}

/// Returns the squared magnitude of the gradient of a variable
/** \param[in] grad The gradient array
 * \param[in] ivar The index of the physical variable whose gradient magnitude is needed
 */
template <typename scalar>
static inline scalar gradientMagnitude2(const Eigen::Array<scalar,NDIM,NVARS>& grad, const int ivar)
{
	scalar res = 0;
	for(int j = 0; j < NDIM; j++)
		res += grad(j,ivar)*grad(j,ivar);
	return res;
}

template <typename scalar>
void WENOReconstruction<scalar>::compute_face_values(const MVector<scalar>& u, 
                                             const amat::Array2d<scalar>& ug,
                                             const GradArray<scalar,NVARS>& grads,
                                             amat::Array2d<scalar>& ufl,
                                             amat::Array2d<scalar>& ufr) const
{
	// first compute limited derivatives at each cell

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			scalar wsum = 0;
			scalar lgrad[NDIM]; 
			zeros(lgrad, NDIM);

			// Central stencil
			const scalar denom = pow( gradientMagnitude2(grads[ielem],ivar) + epsilon , gamma );
			const scalar w = lambda / denom;
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

				const scalar denom = pow( gradientMagnitude2(grads[jelem],ivar) + epsilon , gamma );
				const scalar w = 1.0 / denom;
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

template <typename scalar>
MUSCLReconstruction<scalar>::MUSCLReconstruction(const UMesh2dh<scalar> *const mesh,
		const amat::Array2d<scalar>& r_centres, const amat::Array2d<scalar>* gauss_r)
	: SolutionReconstruction<scalar>(mesh, r_centres, gauss_r), eps{1e-8}, k{1.0/3.0}
{ }

template <typename scalar>
inline scalar MUSCLReconstruction<scalar>::
computeBiasedDifference(const scalar *const ri, const scalar *const rj,
                        const scalar ui, const scalar uj, const scalar *const grads) const
{
	scalar del = 0;
	for(int idim = 0; idim < NDIM; idim++)
		del += grads[idim]*(rj[idim]-ri[idim]);

	return 2.0*del - (uj-ui);
}

template <typename scalar>
inline
scalar MUSCLReconstruction<scalar>::musclReconstructLeft(const scalar ui, const scalar uj, 
			const scalar deltam, const scalar phi) const
{
	return ui + phi/4.0*( (1.0-k*phi)*deltam + (1.0+k*phi)*(uj - ui) );
}

template <typename scalar>
inline
scalar MUSCLReconstruction<scalar>::musclReconstructRight(const scalar ui, const scalar uj, 
			const scalar deltap, const scalar phi) const
{
	return uj - phi/4.0*( (1.0-k*phi)*deltap + (1.0+k*phi)*(uj - ui) );
}

template <typename scalar>
MUSCLVanAlbada<scalar>::MUSCLVanAlbada(const UMesh2dh<scalar> *const mesh,
		const amat::Array2d<scalar>& r_centres, const amat::Array2d<scalar>* gauss_r)
	: MUSCLReconstruction<scalar>(mesh, r_centres, gauss_r)
{ }

template <typename scalar>
void MUSCLVanAlbada<scalar>::compute_face_values(const MVector<scalar>& u, 
                                         const amat::Array2d<scalar>& ug,
                                         const GradArray<scalar,NVARS>& grads,
                                         amat::Array2d<scalar>& ufl,
                                         amat::Array2d<scalar>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_int jelem = m->gintfac(ied,1);

		for(int i = 0; i < NVARS; i++)
		{
			// Note that the copy below is necessary because grads[ielem] is column major
			scalar grad[NDIM];
			for(int j = 0; j < NDIM; j++)
				grad[j] = grads[ielem](j,i);
			
			const scalar deltam = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), ug(ied,i), grad);
			
			scalar phi_l = (2.0*deltam * (ug(ied,i) - u(ielem,i)) + eps) 
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
			scalar gradl[NDIM], gradr[NDIM];
			for(int j = 0; j < NDIM; j++) {
				gradl[j] = grads[ielem](j,i);
				gradr[j] = grads[jelem](j,i);
			}

			const scalar deltam = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), u(jelem,i), gradl);
			const scalar deltap = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), u(jelem,i), gradr);
			
			scalar phi_l = (2.0*deltam * (u(jelem,i) - u(ielem,i)) + eps) 
				/ (deltam*deltam + (u(jelem,i) - u(ielem,i))*(u(jelem,i) - u(ielem,i)) + eps);
			if( phi_l < 0.0) phi_l = 0.0;

			scalar phi_r = (2*deltap * (u(jelem,i) - u(ielem,i)) + eps) 
				/ (deltap*deltap + (u(jelem,i) - u(ielem,i))*(u(jelem,i) - u(ielem,i)) + eps);
			if( phi_r < 0.0) phi_r = 0.0;

			ufl(ied,i) = musclReconstructLeft(u(ielem,i), u(jelem,i), deltam, phi_l);
			ufr(ied,i) = musclReconstructRight(u(ielem,i), u(jelem,i), deltap, phi_r);
		}
	}
}

template <typename scalar>
BarthJespersenLimiter<scalar>::BarthJespersenLimiter(const UMesh2dh<scalar> *const mesh, 
                                             const amat::Array2d<scalar>& r_centres,
                                             const amat::Array2d<scalar>* gauss_r)
	: SolutionReconstruction<scalar>(mesh, r_centres, gauss_r)
{
}

template <typename scalar>
void BarthJespersenLimiter<scalar>::compute_face_values(const MVector<scalar>& u, 
                                                const amat::Array2d<scalar>& ug, 
                                                const GradArray<scalar,NVARS>& grads,
                                                amat::Array2d<scalar>& ufl,
                                                amat::Array2d<scalar>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			scalar duimin=0, duimax=0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int jel = m->gesuel(iel,j);
				const scalar dui = u(jel,ivar)-u(iel,ivar);
				if(dui > duimax) duimax = dui;
				if(dui < duimin) duimin = dui;
			}
			
			scalar lim = 1.0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int face = m->gelemface(iel,j);
				
				const scalar uface = linearExtrapolate(u(iel,ivar), grads[iel], ivar, 1.0,
						&gr[face](0,0), &ri(iel,0));
				
				scalar phiik;
				const scalar diff = uface - u(iel,ivar);
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

template <typename scalar>
VenkatakrishnanLimiter<scalar>
::VenkatakrishnanLimiter(const UMesh2dh<scalar> *const mesh,
                         const amat::Array2d<scalar>& r_centres,
                         const amat::Array2d<scalar>* gauss_r,
                         const a_real k_param)
	: SolutionReconstruction<scalar>(mesh, r_centres, gauss_r), K{k_param}
{
	std::cout << "  Venkatakrishnan Limiter: Constant K = " << K << std::endl;
	// compute characteristic length, currently the maximum edge length, of all cells
	clength.resize(m->gnelem());
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int ifa = 0; ifa < m->gnnode(iel); ifa++)
		{
			scalar llen = 0;
			const int inode = ifa, jnode = (ifa+1) % m->gnnode(iel);
			for(int idim = 0; idim < 2; idim++)
				llen += std::pow(m->gcoords(m->ginpoel(iel,inode),idim) 
						- m->gcoords(m->ginpoel(iel,jnode),idim), 2);

			if(clength[iel] < llen) clength[iel] = llen;
		}
		clength[iel] = std::sqrt(clength[iel]);
	}
}

template <typename scalar>
void VenkatakrishnanLimiter<scalar>
::compute_face_values(const MVector<scalar>& u,
                      const amat::Array2d<scalar>& ug, 
                      const GradArray<scalar,NVARS>& grads,
                      amat::Array2d<scalar>& ufl,
                      amat::Array2d<scalar>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		const scalar eps2 = std::pow(K*clength[iel], 3);

		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			scalar duimin=0, duimax=0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int jel = m->gesuel(iel,j);
				const scalar dui = u(jel,ivar)-u(iel,ivar);
				if(dui > duimax) duimax = dui;
				if(dui < duimin) duimin = dui;
			}
			
			scalar lim = 1.0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const a_int face = m->gelemface(iel,j);
				
				const scalar uface = linearExtrapolate(u(iel,ivar), grads[iel], ivar, 1.0,
						&gr[face](0,0), &ri(iel,0));
				
				const scalar dm = uface - u(iel,ivar);

				// Venkatakrishnan modification
				const scalar dp = dm < 0 ? duimin : duimax;
				const scalar phiik = (dp*dp + 2*dp*dm + eps2)/(dp*dp + dp*dm + 2*dm*dm + eps2);

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

template class LinearUnlimitedReconstruction<a_real>;
template class WENOReconstruction<a_real>;
template class MUSCLVanAlbada<a_real>;
template class BarthJespersenLimiter<a_real>;
template class VenkatakrishnanLimiter<a_real>;

} // end namespace

