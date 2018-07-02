/** @file areconstruction.hpp
 * @brief Slope limiters for variable-extrapolation reconstruction
 * @author Aditya Kashi
 */

#ifndef ARECONSTRUCTION_H
#define ARECONSTRUCTION_H

#include "aconstants.hpp"
#include "utilities/aarray2d.hpp"
#include "mesh/amesh2dh.hpp"

namespace fvens {

/// Abstract class for computing face values from cell-centered values and gradients
/** \note Face values at boundary faces are only computed for the left (interior) side. 
 * Right side values for boundary faces need to computed elsewhere using boundary conditions.
 */
template <typename scalar>
class SolutionReconstruction
{
protected:
	const UMesh2dh<scalar> *const m;            ///< Mesh context
	const amat::Array2d<scalar>& ri;            ///< coords of cell centers of cells
	const amat::Array2d<scalar> *const gr;      ///< coords of Gauss quadrature points of each face
	const int ng;                               ///< Number of Gauss points

public:
	SolutionReconstruction (const UMesh2dh<scalar> *const  mesh,          ///< Mesh context
	                        const amat::Array2d<scalar>& c_centres,       ///< Cell centres
	                        const amat::Array2d<scalar>* gauss_r);        ///< Coords of Gauss points

	virtual void compute_face_values(const MVector<scalar>& unknowns, 
	                                 const amat::Array2d<scalar>& unknow_ghost,
	                                 const GradArray<scalar,NVARS>& grads,
	                                 amat::Array2d<scalar>& uface_left,
	                                 amat::Array2d<scalar>& uface_right) const = 0;

	virtual ~SolutionReconstruction();
};

/// Calculate values of variables at left and right sides of each face 
/// based on computed derivatives but without limiter.
/** ug (cell centered flow variables at ghost cells) are not used for this
 */
template <typename scalar>
class LinearUnlimitedReconstruction : public SolutionReconstruction<scalar>
{
public:
	/// Constructor. \sa SolutionReconstruction::SolutionReconstruction.
	LinearUnlimitedReconstruction(const UMesh2dh<scalar> *const mesh,
	                              const amat::Array2d<scalar>& c_centres, 
	                              const amat::Array2d<scalar>* gauss_r);

	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,NVARS>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;

protected:
	using SolutionReconstruction<scalar>::m;
	using SolutionReconstruction<scalar>::ri;
	using SolutionReconstruction<scalar>::gr;
	using SolutionReconstruction<scalar>::ng;
};

/// Computes state at left and right sides of each face based on WENO-limited derivatives 
/// at each cell
/** References: \cite xia2014, \cite dumbser2007.
 *
 * Note that we do not take the 'oscillation indicator' as the square of the magnitude of 
 * the gradient, like (it seems) in Dumbser & Kaeser, but unlike in Xia et. al.
 */
template <typename scalar>
class WENOReconstruction : public SolutionReconstruction<scalar>
{
	const a_real gamma;               ///< Exponent for oscillation indicator
	const a_real lambda;              ///< Weight of central stencil relative to biased stencils
	const a_real epsilon;             ///< Small constant to avoid division by zero
public:
	WENOReconstruction(const UMesh2dh<scalar> *const mesh,
	                   const amat::Array2d<scalar>& c_centres, 
	                   const amat::Array2d<scalar>* gauss_r,
	                   const a_real central_weight);

	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,NVARS>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar>::m;
	using SolutionReconstruction<scalar>::ri;
	using SolutionReconstruction<scalar>::gr;
	using SolutionReconstruction<scalar>::ng;
};

/// Provides common functionality for computing face values using MUSCL reconstruciton
/** The MUSCL reconstruction for unstructured grids is based on \cite lohner2008.
 */
template <typename scalar>
class MUSCLReconstruction : public SolutionReconstruction<scalar>
{
public:
	MUSCLReconstruction(const UMesh2dh<scalar> *const mesh,
	                    const amat::Array2d<scalar>& c_centres, 
	                    const amat::Array2d<scalar>* gauss_r);
    
	virtual void compute_face_values(const MVector<scalar>& unknowns, 
	                                 const amat::Array2d<scalar>& unknow_ghost, 
	                                 const GradArray<scalar,NVARS>& grads,
	                                 amat::Array2d<scalar>& uface_left,
	                                 amat::Array2d<scalar>& uface_right) const = 0;

protected:
	using SolutionReconstruction<scalar>::m;
	using SolutionReconstruction<scalar>::ri;
	using SolutionReconstruction<scalar>::gr;
	using SolutionReconstruction<scalar>::ng;

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
template <typename scalar>
class MUSCLVanAlbada : public MUSCLReconstruction<scalar>
{
public:
	MUSCLVanAlbada(const UMesh2dh<scalar> *const mesh,
	               const amat::Array2d<scalar>& c_centres, 
	               const amat::Array2d<scalar>* gauss_r);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,NVARS>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar>::m;
	using SolutionReconstruction<scalar>::ri;
	using SolutionReconstruction<scalar>::gr;
	using SolutionReconstruction<scalar>::ng;
	using MUSCLReconstruction<scalar>::computeBiasedDifference;
	using MUSCLReconstruction<scalar>::musclReconstructLeft;
	using MUSCLReconstruction<scalar>::musclReconstructRight;
	using MUSCLReconstruction<scalar>::eps;
	using MUSCLReconstruction<scalar>::k;
};

/// Non-differentiable multidimensional slope limiter for linear reconstruction
template <typename scalar>
class BarthJespersenLimiter : public SolutionReconstruction<scalar>
{
public:
	BarthJespersenLimiter(const UMesh2dh<scalar> *const mesh, 
	                      const amat::Array2d<scalar>& c_centres, 
	                      const amat::Array2d<scalar>* gauss_r);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,NVARS>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar>::m;
	using SolutionReconstruction<scalar>::ri;
	using SolutionReconstruction<scalar>::gr;
	using SolutionReconstruction<scalar>::ng;
};

/// Differentiable modification of Barth-Jespersen limiter
template <typename scalar>
class VenkatakrishnanLimiter: public SolutionReconstruction<scalar>
{
	/// Parameter for adjusting limiting vs convergence
	const a_real K;

	/// List of characteristic length of cells
	std::vector<scalar> clength;

public:
	/** \param[in] k_param Smaller values lead to better limiting at the expense of convergence,
	 *    higher values improve convergence at the expense of some oscillations in the solution.
	 */
	VenkatakrishnanLimiter(const UMesh2dh<scalar> *const mesh, 
	                       const amat::Array2d<scalar>& c_centres, 
	                       const amat::Array2d<scalar>* gauss_r, const a_real k_param);
    
	void compute_face_values(const MVector<scalar>& unknowns, 
	                         const amat::Array2d<scalar>& unknow_ghost, 
	                         const GradArray<scalar,NVARS>& grads,
	                         amat::Array2d<scalar>& uface_left,
	                         amat::Array2d<scalar>& uface_right) const;
protected:
	using SolutionReconstruction<scalar>::m;
	using SolutionReconstruction<scalar>::ri;
	using SolutionReconstruction<scalar>::gr;
	using SolutionReconstruction<scalar>::ng;
};

} // end namespace
#endif
