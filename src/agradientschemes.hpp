/** @file agradientschemes.hpp
 * @brief Classes for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#ifndef AGRADIENTSCHEMES_H
#define AGRADIENTSCHEMES_H 1

#include "amesh2dh.hpp"

namespace acfd
{

/// Abstract class for solution gradient computation schemes
/** For this, we need ghost cell-centered values of flow variables.
 */
class GradientScheme
{
protected:
	const UMesh2dh *const m;                             ///< Mesh context
	const amat::Array2d<a_real> *const rc;               ///< All cell-centres' coordinates

public:
	GradientScheme(const UMesh2dh *const mesh,        ///< Mesh context
			const amat::Array2d<a_real> *const _rc);       ///< Cell centers 
	
	virtual ~GradientScheme();

	/// Computes gradients corresponding to a state vector
	virtual void compute_gradients(
			const MVector *const unk,                         ///< [in] Solution multi-vector
			const amat::Array2d<a_real>*const unkg,           ///< [in] Ghost cell states 
			amat::Array2d<a_real> *const gradx,               ///< [in,out] Gradients in x-direction
			amat::Array2d<a_real> *const grady                ///< [in,out] Gradients in y-direction
		) const = 0;
};

/// Simply sets the gradient to zero
template<short nvars>
class ZeroGradients : public GradientScheme
{
public:
	ZeroGradients(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real> *const unkg, 
			amat::Array2d<a_real> *const gradx, amat::Array2d<a_real> *const grady) const;
};

/**
 * @brief Implements linear reconstruction using the Green-Gauss theorem over elements.
 * 
 * An inverse-distance weighted average is used to obtain the conserved variables at the faces.
 */
template<short nvars>
class GreenGaussGradients : public GradientScheme
{
public:
	GreenGaussGradients(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real> *const unkg, 
			amat::Array2d<a_real> *const gradx, amat::Array2d<a_real> *const grady) const;
};

/// Class implementing linear weighted least-squares reconstruction
template<short nvars>
class WeightedLeastSquaresGradients : public GradientScheme
{
	std::vector<Matrix<a_real,2,2>> V;			///< LHS of least-squares problems
	//Matrix<a_real,2,nvars> d;					///< unknown vector of least-squares problem

public:
	WeightedLeastSquaresGradients(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real> *const unkg, 
			amat::Array2d<a_real> *const gradx, amat::Array2d<a_real> *const grady) const;
};


} // end namespace
#endif
