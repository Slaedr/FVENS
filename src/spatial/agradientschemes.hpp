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
	const UMesh2dh<scalar> *const m;                     ///< Mesh context
	const amat::Array2d<scalar>& rc;                     ///< All cell-centres' coordinates

public:
	GradientScheme(const UMesh2dh<scalar> *const mesh,       ///< Mesh context
	               const amat::Array2d<scalar>& _rc);        ///< Cell centers 
	
	virtual ~GradientScheme();

	/// Computes gradients corresponding to a state vector
	virtual void compute_gradients(
			const MVector<scalar>& unk,                 ///< [in] Solution multi-vector
			const amat::Array2d<scalar>& unkg,          ///< [in] Ghost cell states 
			GradArray<scalar,nvars>& grads ) const = 0;
};

/// Simply sets the gradient to zero
template<typename scalar, int nvars>
class ZeroGradients : public GradientScheme<scalar,nvars>
{
public:
	ZeroGradients(const UMesh2dh<scalar> *const mesh, 
	              const amat::Array2d<scalar>& _rc);

	void compute_gradients(const MVector<scalar>& unk, 
	                       const amat::Array2d<scalar>& unkg, 
	                       GradArray<scalar,nvars>& grads ) const;

protected:
	using GradientScheme<scalar,nvars>::m;
	using GradientScheme<scalar,nvars>::rc;
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
	                    const amat::Array2d<scalar>& _rc);

	void compute_gradients(const MVector<scalar>& unk, 
	                       const amat::Array2d<scalar>& unkg,
	                       GradArray<scalar,nvars>& grads ) const;

protected:
	using GradientScheme<scalar,nvars>::m;
	using GradientScheme<scalar,nvars>::rc;
};

/// Class implementing linear weighted least-squares reconstruction
template<typename scalar, int nvars>
class WeightedLeastSquaresGradients : public GradientScheme<scalar,nvars>
{
public:
	WeightedLeastSquaresGradients(const UMesh2dh<scalar> *const mesh, 
	                              const amat::Array2d<scalar>& _rc);

	void compute_gradients(const MVector<scalar>& unk, 
	                       const amat::Array2d<scalar>& unkg, 
	                       GradArray<scalar,nvars>& grads ) const;

protected:
	using GradientScheme<scalar,nvars>::m;
	using GradientScheme<scalar,nvars>::rc;

private:
	/// The least squares LHS matrix
	DimMatrixArray<scalar> V;
};


} // end namespace
#endif
