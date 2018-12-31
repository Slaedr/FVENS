/** @file agradientschemes.hpp
 * @brief Classes for different gradient estimation schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#ifndef AGRADIENTSCHEMES_H
#define AGRADIENTSCHEMES_H 1

#include "mesh/amesh2dh.hpp"

namespace fvens
{

/// Abstract class for solution gradient computation schemes
/** For this, we need ghost cell-centered values of flow variables.
 */
template<typename scalar, int nvars>
class GradientScheme
{
protected:
	/// Mesh context
	const UMesh2dh<scalar> *const m;
	/// Coords of subdomain and connectivity ghost cell-centres
	const scalar *const rc;
	/// Coords of physical boundary ghost cell centres
	const scalar *const rcbp;

public:
	/// Sets needed data
	/** \param mesh Mesh context
	 * \param _rc Cell centres of subdomain and connectivity ghost cells stored as a row-major
	 *   nElem x ndim array
	 * \param _rcbp Cell centres of physical boundary ghost cells stored as a row-major nElem x ndim
	 *   array
	 */
	GradientScheme(const UMesh2dh<scalar> *const mesh,
	               const scalar *const _rc,
	               const scalar *const _rcbp);
	
	virtual ~GradientScheme();

	/// Computes gradients corresponding to a state vector
	virtual void compute_gradients(
			const MVector<scalar>& unk,              ///< [in] Solution multi-vector
			const amat::Array2dView<scalar> unkg,    ///< [in] Ghost cell states 
			scalar *const grads                      ///< [in,out] Gradients output (pre-allocated)
	                               ) const = 0;
};

/// Simply sets the gradient to zero
template<typename scalar, int nvars>
class ZeroGradients : public GradientScheme<scalar,nvars>
{
public:
	ZeroGradients(const UMesh2dh<scalar> *const mesh, 
	              const scalar *const _rc,
	              const scalar *const _rcbp);

	void compute_gradients(const MVector<scalar>& unk, 
	                       const amat::Array2dView<scalar> unkg, 
	                       scalar *const grads ) const;

protected:
	using GradientScheme<scalar,nvars>::m;
	using GradientScheme<scalar,nvars>::rc;
	using GradientScheme<scalar,nvars>::rcbp;
};

/**
 * @brief Implements linear reconstruction using the Green-Gauss theorem over elements.
 * 
 * An inverse-distance weighted average is used to obtain the conserved variables at the faces.
 */
template<typename scalar, int nvars>
class GreenGaussGradients : public GradientScheme<scalar,nvars>
{
public:
	GreenGaussGradients(const UMesh2dh<scalar> *const mesh, 
	                    const scalar *const _rc,
	                    const scalar *const _rcbp);

	void compute_gradients(const MVector<scalar>& unk, 
	                       const amat::Array2dView<scalar> unkg,
	                       scalar *const grads ) const;

protected:
	using GradientScheme<scalar,nvars>::m;
	using GradientScheme<scalar,nvars>::rc;
	using GradientScheme<scalar,nvars>::rcbp;
};

/// Class implementing linear weighted least-squares reconstruction
template<typename scalar, int nvars>
class WeightedLeastSquaresGradients : public GradientScheme<scalar,nvars>
{
public:
	WeightedLeastSquaresGradients(const UMesh2dh<scalar> *const mesh, 
	                              const scalar *const _rc,
	                              const scalar *const _rcbp);

	void compute_gradients(const MVector<scalar>& unk, 
	                       const amat::Array2dView<scalar> unkg, 
	                       scalar *const grads ) const;

protected:
	using GradientScheme<scalar,nvars>::m;
	using GradientScheme<scalar,nvars>::rc;
	using GradientScheme<scalar,nvars>::rcbp;

private:
	/// The least squares LHS matrix
	DimMatrixArray<scalar> V;
};


} // end namespace
#endif
