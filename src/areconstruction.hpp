/** @file areconstruction.hpp
 * @brief Classes for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#ifndef __AMESH2DH_H
#include <amesh2dh.hpp>
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
	const amat::Matrix<acfd_real>* rc;
	/// Ghost cell centers
	const amat::Matrix<acfd_real>* rcg;
	/// Cell-centered flow vaiables
	const amat::Matrix<acfd_real>* u;
	/// flow variables at ghost cells
	const amat::Matrix<acfd_real>* ug;
	/// Cell-centred x-gradients
	amat::Matrix<acfd_real>* dudx;
	/// Cell-centred y-gradients
	amat::Matrix<acfd_real>* dudy;

public:
	virtual ~Reconstruction();
	virtual void setup(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unk, const amat::Matrix<acfd_real>* unkg, amat::Matrix<acfd_real>* gradx, amat::Matrix<acfd_real>* grady, 
			const amat::Matrix<acfd_real>* _rc, const amat::Matrix<acfd_real>* const _rcg);
	virtual void compute_gradients() = 0;
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
	std::vector<amat::Matrix<acfd_real>> V;		///< LHS of least-squares problem
	std::vector<amat::Matrix<acfd_real>> f;		///< RHS of least-squares problem
	amat::Matrix<acfd_real> d;					///< unknown vector of least-squares problem
	amat::Matrix<acfd_real> idets;				///< inverse of determinants of the LHS
	amat::Matrix<acfd_real> du;

public:
	void setup(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unk, const amat::Matrix<acfd_real>* unkg, amat::Matrix<acfd_real>* gradx, amat::Matrix<acfd_real>* grady, 
			const amat::Matrix<acfd_real>* _rc, const amat::Matrix<acfd_real>* const _rcg);
	void compute_gradients();
};


} // end namespace
