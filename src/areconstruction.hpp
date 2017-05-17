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
	/// Ghost cell centers
	const amat::Array2d<a_real>* rcg;
	/// Cell-centered flow vaiables
	const Matrix* u;
	/// flow variables at ghost cells
	const amat::Array2d<a_real>* ug;
	/// Cell-centred x-gradients
	amat::Array2d<a_real>* dudx;
	/// Cell-centred y-gradients
	amat::Array2d<a_real>* dudy;

public:
	virtual ~Reconstruction();
	virtual void setup(const UMesh2dh* mesh, const Matrix* unk, const amat::Array2d<a_real>* unkg, amat::Array2d<a_real>* gradx, amat::Array2d<a_real>* grady, 
			const amat::Array2d<a_real>* _rc, const amat::Array2d<a_real>* const _rcg);
	virtual void compute_gradients() = 0;
};

/// Simply sets the gradient to zero
class ConstantReconstruction : public Reconstruction
{
public:
	void compute_gradients();
};

/**
 * @brief Implements linear reconstruction using the Green-Gauss theorem over elements.
 * 
 * An inverse-distance weighted average is used to obtain the conserved variables at the faces.
 */
class GreenGaussReconstruction : public Reconstruction
{
public:
	void compute_gradients();
};


/// Class implementing linear weighted least-squares reconstruction
class WeightedLeastSquaresReconstruction : public Reconstruction
{
	std::vector<amat::Array2d<a_real>> V;		///< LHS of least-squares problem
	std::vector<amat::Array2d<a_real>> f;		///< RHS of least-squares problem
	amat::Array2d<a_real> d;					///< unknown vector of least-squares problem
	amat::Array2d<a_real> idets;				///< inverse of determinants of the LHS
	amat::Array2d<a_real> du;

public:
	void setup(const UMesh2dh* mesh, const Matrix* unk, const amat::Array2d<a_real>* unkg, amat::Array2d<a_real>* gradx, amat::Array2d<a_real>* grady, 
			const amat::Array2d<a_real>* _rc, const amat::Array2d<a_real>* const _rcg);
	void compute_gradients();
};


} // end namespace
