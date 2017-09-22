/** @file areconstruction.hpp
 * @brief Classes for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#ifndef __AMESH2DH_H
#include "amesh2dh.hpp"
#endif

#define __ARECONSTRUCTION_H 1

namespace acfd
{

/// Abstract class for variable gradient reconstruction schemes
/** For this, we need ghost cell-centered values of flow variables.
 */
class Reconstruction
{
protected:
	const UMesh2dh* m;
	/// Cell centers' coords
	const amat::Array2d<a_real>* rc;

public:
	/// Base constructor
	Reconstruction(const UMesh2dh *const mesh,             ///< Mesh context
			const amat::Array2d<a_real> *const _rc);       ///< Cell centers 
	
	virtual ~Reconstruction();

	virtual void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor>*const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady) = 0;
};

/// Simply sets the gradient to zero
template<short nvars>
class ConstantReconstruction : public Reconstruction
{
public:
	ConstantReconstruction(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor>*const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady);
};

/**
 * @brief Implements linear reconstruction using the Green-Gauss theorem over elements.
 * 
 * An inverse-distance weighted average is used to obtain the conserved variables at the faces.
 */
template<short nvars>
class GreenGaussReconstruction : public Reconstruction
{
public:
	GreenGaussReconstruction(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady);
};


/// Class implementing linear weighted least-squares reconstruction
template<short nvars>
class WeightedLeastSquaresReconstruction : public Reconstruction
{
	std::vector<Matrix<a_real,2,2>> V;			///< LHS of least-squares problems
	std::vector<Matrix<a_real,2,nvars>> f;		///< RHS of least-squares problems
	//Matrix<a_real,2,nvars> d;					///< unknown vector of least-squares problem

public:
	WeightedLeastSquaresReconstruction(const UMesh2dh *const mesh, 
			const amat::Array2d<a_real> *const _rc);

	void compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const unk, 
			const amat::Array2d<a_real>*const unkg, 
			amat::Array2d<a_real>*const gradx, amat::Array2d<a_real>*const grady);
};


} // end namespace
