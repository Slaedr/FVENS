#ifndef __ALIMITER_H

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX_H
#include <amatrix.hpp>
#endif

#ifndef __AMESH2DH_H
#include <amesh2dh.hpp>
#endif

#define __ALIMITER_H

namespace acfd {

/// Abstract class for computing face values (for each Gauss point) from cell-centered values and gradients
/** \note Face values at boundary faces are only computed for the left (interior) side. Right side values for boundary faces need to computed elsewhere using boundary conditions.
 */
class FaceDataComputation
{
protected:
	const UMesh2dh* m;
	const amat::Matrix<acfd_real>* u;			///< Cell-centered flow data
	const amat::Matrix<acfd_real>* ug;			///< Cell-centered ghost cell flow data
	const amat::Matrix<acfd_real>* dudx;
	const amat::Matrix<acfd_real>* dudy;
	const amat::Matrix<acfd_real>* rb;			///< coords of cell centers of ghost cells
	const amat::Matrix<acfd_real>* ri;			///< coords of cell centers of real cells
	const amat::Matrix<acfd_real>* gr;		/// coords of gauss points of each face
	amat::Matrix<acfd_real>* ufl;			///< left face flow data
	amat::Matrix<acfd_real>* ufr;			///< right face flow data
	int ng;									///< Number of Gauss points

public:
	FaceDataComputation();
    FaceDataComputation (const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
			const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
			const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, 
			const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
    void setup(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
			const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, 
			const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
	virtual void compute_face_values() = 0;
	virtual ~FaceDataComputation();
};

/// Calculate values of variables at left and right sides of each face based on computed derivatives but without limiter.
/** ug (cell centered flow variables at ghost cells) are not used for this
 */
class NoLimiter : public FaceDataComputation
{
public:
	/// Constructs the NoLimiter object. \sa FaceDataComputation::FaceDataComputation.
	NoLimiter(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
			const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
			const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, 
			const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
	void compute_face_values();
};

/// Computes state at left and right sides of each face based on WENO-limited derivatives at each cell
/** References:
 * - Y. Xia, X. Liu and H. Luo. "A finite volume method based on a WENO reconstruction for compressible flows on hybrid grids", 52nd AIAA Aerospace Sciences Meeting, AIAA-2014-0939.
 * - M. Dumbser and M. Kaeser. "Arbitrary high order non-oscillatory finite volume schemes on unsttructured meshes for linear hyperbolic systems", J. Comput. Phys. 221 pp 693--723, 2007.
 *
 * Note that we do not take the 'oscillation indicator' as the square of the magnitude of the gradient, like (it seems) in Dumbser & Kaeser, but unlike in Xia et. al.
 */
class WENOLimiter : public FaceDataComputation
{
	amat::Matrix<acfd_real>* ldudx;
	amat::Matrix<acfd_real>* ldudy;
	acfd_real gamma;
	acfd_real lambda;
	acfd_real epsilon;
public:
    WENOLimiter(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
			const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, const amat::Matrix<acfd_real>* gauss_r, 
			amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
	void compute_face_values();
	~WENOLimiter();
};

/// Computes face values using the Van-Albada limiter
class VanAlbadaLimiter : public FaceDataComputation
{
    acfd_real eps;
    acfd_real k;               			/// van-Albada parameter
	amat::Matrix<acfd_real> phi_l;		/// left-face limiter values
	amat::Matrix<acfd_real> phi_r;		/// right-face limiter values

public:
	void setup(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* r_centres,
		const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
    
	/// Calculate values of variables at left and right sides of each face based on computed derivatives and limiter values
	void compute_face_values();
	// Computes face values using my own simplistic limiting (probably bogus)
    //void compute_reg_face_values();
};

/*class BarthJaspersonLimiter
{
    UMesh2dh* m;
    Matrix<double>* u;
    Matrix<double>* dudx;
    Matrix<double>* dudy;
    Matrix<double>* gaussx;
    Matrix<double>* gaussy;
    Matrix<double>* xi;
    Matrix<double>* yi;
    int nvars;
    int ngauss;

    Matrix<double>* phi_l;
    Matrix<double>* phi_r;

    Matrix<double> umax;        // holds, for each element, max value of u over all its face-neighbors
    Matrix<double> umin;        // holds, for each element, min value of u over all its face-neighbors

    void compute_maxmin()
    {
        for(int iel = 0; iel < m->gnelem(); iel++)
        {
            for(int i = 0; i < nvars; i++)
            {
                umax(iel,i) = u->get(m->gesuel(iel,0),i);
                umin(iel,i) = u->get(m->gesuel(iel,0),i);
            }

            for(int iface = 1; iface < m->gnfael(iel); iface++)
            {
                for(int i = 0; i < nvars; i++)
                {
                    if(u->get(m->gesuel(iel,iface),i) > umax(iel,i)) umax(iel,i) = u->get(m->gesuel(iel,iface),i);
                    if(u->get(m->gesuel(iel,iface),i) > umin(iel,i)) umin(iel,i) = u->get(m->gesuel(iel,iface),i);
                }
            }
        }
    }

public:
    BarthJaspersonLimiter() {}

    void setup_limiter(UMesh2dh* mesh, Matrix<double>* unknowns, Matrix<double>* x_deriv, Matrix<double>* y_deriv, Matrix<double>* x_centres, Matrix<double>* y_centres, Matrix<double>* x_gauss, Matrix<double>* y_gauss, Matrix<double>* limits_l, Matrix<double>* limits_r)
    {
        m = mesh;
        u = unknowns;
        nvars = u->cols();
        dudx = x_deriv;
        dudy = y_deriv;
        xi = x_centres;
        yi = y_centres;
        gaussx = x_gauss;
        gaussy = y_gauss;
        phi_l = limits_l;
        phi_r = limits_r;

        ngauss = gaussx->cols();

        umax.setup(m->gnelem(), nvars, ROWMAJOR);
        umin.setup(m->gnelem(), nvars, ROWMAJOR);
    }

    void compute_limiters()
    {
        compute_maxmin();

        //Matrix<double>* phig;       // one matrix for each element - each matrix stores phi_{i,g} for each gauss point g and for all 4 variables
        //phig = new Matrix<double>[m->gnelem()];
        //for(int iel = 0; iel < m->gnelem(); iel++)
        //	phig[iel].setup(ngauss,nvars,ROWMAJOR);

        for(int ied = 0; ied < m->gnbface(); ied++)
        {
            int lel = m->gintfac(ied,0);
            Matrix<double> ugminus(ngauss,nvars,ROWMAJOR);
            Matrix<double> ugplus(ngauss, nvars, ROWMAJOR);
            Matrix<double> phig_l(ngauss,nvars,ROWMAJOR);
            Matrix<double> phig_r(ngauss,nvars,ROWMAJOR);
            for(int g = 0; g < ngauss; g++)
            {
                for(int i = 0; i < nvars; i++)
                {
                    ugminus(g,i) = dudx->get(lel,i)*(gaussx->get(ied,g)-xi->get(lel)) + dudy->get(lel,i)*(gaussy->get(ied,g)-yi->get(lel));
                    ugplus(g,i) = (ugminus(g,i) > 0) ? umax(lel,i) - u->get(lel,i) : umin(lel,i) - u->get(lel,i);
                    //phig_l
                }
            }
        }

        for(int ied = m->gnbface(); ied < m->gnaface(); ied++)
        {
            int lel = m->gintfac(ied,0);	// left element
			int rel = m->gintfac(ied,1);	// right element
        }
    }
};*/

} // end namespace
#endif
