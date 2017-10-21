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

/// Abstract class for variable gradient reconstruction schemes
/** For this, we need ghost cell-centered values of flow variables.
 */
class GradientComputation
{
protected:
	const UMesh2dh* m;
	/// Cell centers' coords
	const amat::Array2d<a_real>* rc;

public:
	/// Base constructor
	GradientComputation(const UMesh2dh *const mesh,             ///< Mesh context
			const amat::Array2d<a_real> *const _rc);       ///< Cell centers 
	
	virtual ~GradientComputation();

	virtual void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor>*const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady) const = 0;
};

/// Simply sets the gradient to zero
template<short nvars>
class ZeroGradients : public GradientComputation
{
public:
	ZeroGradients(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor>*const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady) const;
};

/**
 * @brief Implements linear reconstruction using the Green-Gauss theorem over elements.
 * 
 * An inverse-distance weighted average is used to obtain the conserved variables at the faces.
 */
template<short nvars>
class GreenGaussGradients : public GradientComputation
{
public:
	GreenGaussGradients(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady) const;
};

/// Class implementing linear weighted least-squares reconstruction
template<short nvars>
class WeightedLeastSquaresGradients : public GradientComputation
{
	std::vector<Matrix<a_real,2,2>> V;			///< LHS of least-squares problems
	//Matrix<a_real,2,nvars> d;					///< unknown vector of least-squares problem

public:
	WeightedLeastSquaresGradients(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady) const;
};


} // end namespace
#endif
