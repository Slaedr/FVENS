#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#define __ALIMITER_H

namespace acfd {

/// Abstract class for computing face values (for each Gauss point) from cell-centered values and gradients
/** \note Face values at boundary faces are only computed for the left (interior) side. Right side values for boundary faces need to computed elsewhere using boundary conditions.
 */
class FaceDataComputation
{
protected:
	const UTriMesh* m;
	const amat::Matrix<acfd_real>* u;			///< Cell-centered flow data
	const amat::Matrix<acfd_real>* ug;			///< Cell-centered ghost cell flow data
	const amat::Matrix<acfd_real>* dudx;
	const amat::Matrix<acfd_real>* dudy;
	const amat::Matrix<acfd_real>* xb;
	const amat::Matrix<acfd_real>* yb;
	const amat::Matrix<acfd_real>* xi;
	const amat::Matrix<acfd_real>* yi;
	const amat::Matrix<acfd_real>* gx;		/// x-coords of gauss points of each face
	const amat::Matrix<acfd_real>* gy;		/// y-coordinates of gauss points of each face
	amat::Matrix<acfd_real>* ufl;			///< left face flow data
	amat::Matrix<acfd_real>* ufr;			///< right face flow data
	int nvars;								///< Number of flow variables
	int ng;									///< Number of Gauss points
	
	// for local use
	int ielem;
	int jelem;
	int ied;
	acfd_real nx;
	acfd_real ny;

public:
    void setup(const UTriMesh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
			const amat::Matrix<acfd_real>* x_ghost_centres, const amat::Matrix<acfd_real>* y_ghost_centres, const amat::Matrix<acfd_real>* x_centres, const amat::Matrix<acfd_real>* y_centres, 
			const amat::Matrix<acfd_real>* gauss_x, const amat::Matrix<acfd_real>* gauss_y, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
	virtual void compute_face_values() = 0;
};

void FaceDataComputation::setup(const UTriMesh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* x_ghost_centres, const amat::Matrix<acfd_real>* y_ghost_centres, const amat::Matrix<acfd_real>* x_centres, const amat::Matrix<acfd_real>* y_centres, 
		const amat::Matrix<acfd_real>* gauss_x, const amat::Matrix<acfd_real>* gauss_y, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
{
	m = mesh;
	u = unknowns;
	ug = unknow_ghost;          // contains ghost cell states according to BCs for each boundary edge
	nvars = u->cols();
	dudx = x_deriv;
	dudy = y_deriv;
	xi = x_centres;
	yi = y_centres;
	xb = x_ghost_centres;       // contains x-coord of right "cell centroid" of each boundary edge
	yb = y_ghost_centres;       // contains y-coord of right "cell centroid" of each boundary edge
	gx = gauss_x;
	gy = gauss_y;
	ufl = uface_left;
	ufr = uface_right;
	ng = gx->cols();
}

/// Calculate values of variables at left and right sides of each face based on computed derivatives but without limiter.
/** ug (cell centered flow variables at ghost cells) are not used for this
 */
class NoLimiter : public FaceDataComputation
{
public:
	void compute_face_values();
};

void NoLimiter::compute_face_values()
{
	// (a) internal faces
	//cout << "NoLimiter: compute_face_values(): Computing values at faces - internal\n";
	for(ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		ielem = m->gintfac(ied,0);
		jelem = m->gintfac(ied,1);

		//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
		for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
		{
			for(int i = 0; i < nvars; i++)
			{

				(*ufl)(ied,i) = u->get(ielem,i) + dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));
				(*ufr)(ied,i) = u->get(jelem,i) + dudx->get(jelem,i)*(gx->get(ied,0)-xi->get(jelem)) + dudy->get(jelem,i)*(gy->get(ied,0)-yi->get(jelem));
			}
		}
	}
	//cout << "NoLimiter: compute_unlimited_interface_values(): Computing values at faces - boundary\n";
	//Now calculate ghost states at boundary faces using the ufl and ufr of cells
	for(ied = 0; ied < m->gnbface(); ied++)
	{
		ielem = m->gintfac(ied,0);
		nx = m->ggallfa(ied,0);
		ny = m->ggallfa(ied,1);

		for(int ig = 0; ig < ng; ig++)
		{
			for(int i = 0; i < nvars; i++)
				(*ufl)(ied,i) = u->get(ielem,i) + dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));
		}
	}
}

/// Computes face values using the Van-Albada limiter
class VanAlbadaLimiter : public FaceDataComputation
{
    acfd_real eps;
    acfd_real k;               			/// van-Albada parameter
	amat::Matrix<acfd_real> phi_l;		/// left-face limiter values
	amat::Matrix<acfd_real> phi_r;		/// right-face limiter values

public:
	void setup(const UTriMesh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* x_ghost_centres, const amat::Matrix<acfd_real>* y_ghost_centres, const amat::Matrix<acfd_real>* x_centres, const amat::Matrix<acfd_real>* y_centres, 
		const amat::Matrix<acfd_real>* gauss_x, const amat::Matrix<acfd_real>* gauss_y, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right);
    void compute_limiters();
    /// Calculate values of variables at left and right sides of each face based on computed derivatives and limiter values
	void compute_face_values();
	/// Computes face values using my own simplistic limiting (probably bogus)
    void compute_reg_face_values();
};

void VanAlbadaLimiter::setup(const UTriMesh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
	const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
	const amat::Matrix<acfd_real>* x_ghost_centres, const amat::Matrix<acfd_real>* y_ghost_centres, const amat::Matrix<acfd_real>* x_centres, const amat::Matrix<acfd_real>* y_centres, 
	const amat::Matrix<acfd_real>* gauss_x, const amat::Matrix<acfd_real>* gauss_y, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
{
	FaceDataComputation::setup(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, x_ghost_centres, y_ghost_centres, x_centres, y_centres, 
		gauss_x, gauss_y, uface_left, uface_right);
	eps = 1e-8;
	k = 1.0/3.0;
	phi_l.setup(m->gnaface(), nvars);
	phi_r.setup(m->gnaface(), nvars);
}

void VanAlbadaLimiter::compute_limiters()
{
	int i, lel, rel;
	for(ied = 0; ied < m->gnbface(); ied++)
	{
		int lel = m->gintfac(ied,0);
		amat::Matrix<acfd_real> deltam(nvars,1);
		for(i = 0; i < nvars; i++)
		{
			deltam(i) = 2 * ( dudx->get(lel,i)*(xb->get(ied)-xi->get(lel)) + dudy->get(lel,i)*(yb->get(ied)-yi->get(lel)) ) - (ug->get(ied,i) - u->get(lel,i));
			phi_l(ied,i) = (2*deltam(i) * (ug->get(ied,i) - u->get(lel,i)) + eps) / (deltam(i)*deltam(i) + (ug->get(ied,i) - u->get(lel,i))*(ug->get(ied,i) - u->get(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;
		}
	}

	for(ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		lel = m->gintfac(ied,0);
		rel = m->gintfac(ied,1);
		amat::Matrix<acfd_real> deltam(nvars,1);
		amat::Matrix<acfd_real> deltap(nvars,1);
		for(i = 0; i < nvars; i++)
		{
			deltam(i) = 2 * ( dudx->get(lel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(lel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));
			deltap(i) = 2 * ( dudx->get(rel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(rel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));

			phi_l(ied,i) = (2*deltam(i) * (u->get(rel,i) - u->get(lel,i)) + eps) / (deltam(i)*deltam(i) + (u->get(rel,i) - u->get(lel,i))*(u->get(rel,i) - u->get(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;

			phi_r(ied,i) = (2*deltap(i) * (u->get(rel,i) - u->get(lel,i)) + eps) / (deltap(i)*deltap(i) + (u->get(rel,i) - u->get(lel,i))*(u->get(rel,i) - u->get(lel,i)) + eps);
			if( phi_r(ied,i) < 0.0) phi_r(ied,i) = 0.0;
		}
	}
}

/// Calculate values of variables at left and right sides of each face based on computed derivatives and limiter values
void VanAlbadaLimiter::compute_face_values()
{
	compute_limiters();

	int lel, rel, i, ig;
	//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - internal\n";
	amat::Matrix<acfd_real> deltam(nvars,1);
	amat::Matrix<acfd_real> deltap(nvars,1);
	for(ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		ielem = m->gintfac(ied,0); lel = ielem;
		jelem = m->gintfac(ied,1); rel = jelem;

		// NOTE: Only for 1 Gauss point per face
		//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
		for(ig = 0; ig < ng; ig++)      // iterate over gauss points
		{
			for(i = 0; i < nvars; i++)
			{
				deltam(i) = 2 * ( dudx->get(lel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(lel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));
				deltap(i) = 2 * ( dudx->get(rel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(rel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));

				(*ufl)(ied,i) = u->get(ielem,i) + phi_l.get(ied,i)/4.0*( (1-k*phi_l.get(ied,i))*deltam.get(i) + (1+k*phi_l.get(ied,i))*(u->get(rel,i) - u->get(lel,i)) );
				(*ufr)(ied,i) = u->get(jelem,i) + phi_r.get(ied,i)/4.0*( (1-k*phi_r.get(ied,i))*deltap(i) + (1+k*phi_r.get(ied,i))*(u->get(rel,i) - u->get(lel,i)) );
			}
		}
	}
}

void VanAlbadaLimiter::compute_reg_face_values()
{
	// Calculate values of variables at left and right sides of each face based on computed derivatives
	// (a) internal faces
	//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - internal\n";
	for(int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		int ielem = m->gintfac(ied,0); int lel = ielem;
		int jelem = m->gintfac(ied,1); int rel = jelem;

		// TODO: correct for multiple gauss points
		//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
		for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
		{
			for(int i = 0; i < nvars; i++)
			{

				(*ufl)(ied,i) = u->get(ielem,i) + phi_l.get(ied,i)*dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + phi_l.get(ied,i)*dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));
				(*ufr)(ied,i) = u->get(jelem,i) + phi_r.get(ied,i)*dudx->get(jelem,i)*(gx->get(ied,0)-xi->get(jelem)) + phi_r.get(ied,i)*dudy->get(jelem,i)*(gy->get(ied,0)-yi->get(jelem));
			}
		}
	}
	//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - boundary\n";
	//Now calculate ghost states at boundary faces using the ufl and ufr of cells
	// (b) boundary faces
	for(int ied = 0; ied < m->gnbface(); ied++)
	{
		int ielem = m->gintfac(ied,0);
		double nx = m->ggallfa(ied,0);
		double ny = m->ggallfa(ied,1);

		for(int ig = 0; ig < ng; ig++)
		{
			for(int i = 0; i < nvars; i++)
				(*ufl)(ied,i) = u->get(ielem,i) + phi_l.get(ied,i)*dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + phi_l.get(ied,i)*dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));

		}
	}
}

/*class BarthJaspersonLimiter
{
    UTriMesh* m;
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

            for(int iface = 1; iface < m->gnfael(); iface++)
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

    void setup_limiter(UTriMesh* mesh, Matrix<double>* unknowns, Matrix<double>* x_deriv, Matrix<double>* y_deriv, Matrix<double>* x_centres, Matrix<double>* y_centres, Matrix<double>* x_gauss, Matrix<double>* y_gauss, Matrix<double>* limits_l, Matrix<double>* limits_r)
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
