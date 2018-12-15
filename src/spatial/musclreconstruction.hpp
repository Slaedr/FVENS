/** \file
 * \file MUSCL reconstuction schemes
 * \author Aditya Kashi
 */

#ifndef FVENS_MUSCL_RECONSTRUCTION_H
#define FVENS_MUSCL_RECONSTRUCTION_H

#include "areconstruction.hpp"

namespace fvens {

/// Provides common functionality for computing face values using MUSCL reconstruciton
/** The MUSCL reconstruction for unstructured grids is based on \cite lohner2008.
 */
template <typename scalar, int nvars>
class MUSCLReconstruction : public SolutionReconstruction<scalar,nvars>
{
public:
	MUSCLReconstruction(const UMesh2dh<scalar> *const mesh,
	                    const scalar *const c_centres, 
	                    const scalar *const c_centres_ghost,
	                    const amat::Array2d<scalar>& gauss_r);
    
	virtual void compute_face_values(const MVector<scalar>& unknowns, 
	                                 const amat::Array2d<scalar>& unknow_ghost, 
	                                 const GradBlock_t<scalar,NDIM,nvars> *const grads,
	                                 amat::Array2d<scalar>& uface_left,
	                                 amat::Array2d<scalar>& uface_right) const = 0;

protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::ribp;
	using SolutionReconstruction<scalar,nvars>::gr;

	const a_real eps;                       ///< Small number
	const a_real k;                         ///< MUSCL order parameter

	/// Computes a biased difference
	/** The direction of biasing depends on the gradients supplied in the last parameter.
	 * If the gradient of the left cell is given, the backward-biased difference is computed;
	 * if the gradient of the right cell is given, the forward-biased difference is computed.
	 */
	scalar computeBiasedDifference(const scalar *const ri, const scalar *const rj,
	                               const scalar ui, const scalar uj,
	                               const scalar *const grads) const;

	/// Computes the MUSCL reconstructed face value on the left, given the limiter value
	/** \param ui Left cell-centred value
	 * \param uj Right cell-centred value
	 * \param deltam The backward-biased difference ie, delta minus, the analog of u_i - u_(i-1)
	 * \param phi The limiter value
	 * \return The left state at the face between cells i and j
	 */
	scalar musclReconstructLeft(const scalar ui, const scalar uj, 
	                            const scalar deltam, const scalar phi) const;
	
	/// Computes the MUSCL reconstructed face value on the right, given the limiter value
	/** \param ui Left cell-centred value
	 * \param uj Right cell-centred value
	 * \param deltap The forward-biased difference ie, delta plus, the analog of u_(j+1) - u_j
	 * \param phi The limiter value
	 * \return The right state at the face between cells i and j
	 */
	scalar musclReconstructRight(const scalar ui, const scalar uj, 
	                             const scalar deltap, const scalar phi) const;
};

/// Computes face values using MUSCL reconstruciton with Van-Albada limiter
template <typename scalar, int nvars>
class MUSCLVanAlbada : public MUSCLReconstruction<scalar,nvars>
{
public:
	MUSCLVanAlbada(const UMesh2dh<scalar> *const mesh,
	               const scalar *const c_centres, 
	               const scalar *const c_centres_ghost,
	               const amat::Array2d<scalar>& gauss_r);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradBlock_t<scalar,NDIM,nvars> *const grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar,nvars>::m;
	using SolutionReconstruction<scalar,nvars>::ri;
	using SolutionReconstruction<scalar,nvars>::ribp;
	using SolutionReconstruction<scalar,nvars>::gr;
	using MUSCLReconstruction<scalar,nvars>::computeBiasedDifference;
	using MUSCLReconstruction<scalar,nvars>::musclReconstructLeft;
	using MUSCLReconstruction<scalar,nvars>::musclReconstructRight;
	using MUSCLReconstruction<scalar,nvars>::eps;
	using MUSCLReconstruction<scalar,nvars>::k;
};

}

#endif
