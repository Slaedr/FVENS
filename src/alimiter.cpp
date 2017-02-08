#include <alimiter.hpp>

namespace acfd {

FaceDataComputation::FaceDataComputation()
{ }

FaceDataComputation::FaceDataComputation (const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, 
		const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
	:
	m(mesh),
	u (unknowns),
	ug (unknow_ghost),          // contains ghost cell states according to BCs for each boundary edge
	dudx (x_deriv),
	dudy (y_deriv),
	ri (c_centres),
	rb (ghost_centres),       // contains coords of right "cell centroid" of each boundary edge
	gr (gauss_r),
	ufl (uface_left),
	ufr (uface_right),
	ng (gauss_r[0].rows())
{ }

FaceDataComputation::~FaceDataComputation()
{ }

void FaceDataComputation::setup(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres,
		const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
{
	m = mesh;
	u = unknowns;
	ug = unknow_ghost;          // contains ghost cell states according to BCs for each boundary edge
	dudx = x_deriv;
	dudy = y_deriv;
	ri = c_centres;
	rb = ghost_centres;       // contains x-coord of right "cell centroid" of each boundary edge
	gr = gauss_r;
	ufl = uface_left;
	ufr = uface_right;
	ng = gr[0].rows();
}

NoLimiter::NoLimiter(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, 
		const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
	: FaceDataComputation(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, ghost_centres, c_centres, gauss_r, uface_left, uface_right)
{ }

void NoLimiter::compute_face_values()
{
	// (a) internal faces
	//cout << "NoLimiter: compute_face_values(): Computing values at faces - internal\n";
#pragma omp parallel default(shared)
	{
#pragma omp for
		for(acfd_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			acfd_int ielem = m->gintfac(ied,0);
			acfd_int jelem = m->gintfac(ied,1);

			//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
			for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
			{
				for(int i = 0; i < NVARS; i++)
				{

					(*ufl)(ied,i) = u->get(ielem,i) + dudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + dudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
					(*ufr)(ied,i) = u->get(jelem,i) + dudx->get(jelem,i)*(gr[ied].get(ig,0)-ri->get(jelem,0)) + dudy->get(jelem,i)*(gr[ied].get(ig,1)-ri->get(jelem,1));
				}
			}
		}
		//cout << "NoLimiter: compute_unlimited_interface_values(): Computing values at faces - boundary\n";
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
#pragma omp for
		for(acfd_int ied = 0; ied < m->gnbface(); ied++)
		{
			acfd_int ielem = m->gintfac(ied,0);

			for(int ig = 0; ig < ng; ig++)
			{
				for(int i = 0; i < NVARS; i++)
					(*ufl)(ied,i) = u->get(ielem,i) + dudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + dudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
			}
		}
	}
}

WENOLimiter::WENOLimiter(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
		const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
		const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* c_centres, const amat::Matrix<acfd_real>* gauss_r, 
		amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
	: FaceDataComputation(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, ghost_centres, c_centres, gauss_r, uface_left, uface_right)
{
	ldudx = new amat::Matrix<acfd_real>(m->gnelem(),NVARS);
	ldudy = new amat::Matrix<acfd_real>(m->gnelem(),NVARS);
	// values below chosen from second reference (Dumbser and Kaeser)
	gamma = 4.0;
	lambda = 1e3;
	epsilon = 1e-5;
}

WENOLimiter::~WENOLimiter()
{
	delete ldudx;
	delete ldudy;
}

void WENOLimiter::compute_face_values()
{
	// first compute limited derivatives at each cell

#pragma omp parallel default(shared)
	{
#pragma omp for
		for(acfd_int ielem = 0; ielem < m->gnelem(); ielem++)
		{
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				acfd_real wsum = 0;
				(*ldudx)(ielem,ivar) = 0;
				(*ldudy)(ielem,ivar) = 0;

				// Central stencil
				acfd_real denom = pow( dudx->get(ielem,ivar)*dudx->get(ielem,ivar) + dudy->get(ielem,ivar)*dudy->get(ielem,ivar) + epsilon, gamma);
				acfd_real w = lambda / denom;
				wsum += w;
				(*ldudx)(ielem,ivar) += w*dudx->get(ielem,ivar);
				(*ldudy)(ielem,ivar) += w*dudy->get(ielem,ivar);

				// Biased stencils
				for(int jel = 0; jel < m->gnfael(ielem); jel++)
				{
					acfd_int jelem = m->gesuel(ielem,jel);

					// ignore ghost cells
					if(jelem >= m->gnelem())
						continue;

					denom = pow( dudx->get(jelem,ivar)*dudx->get(jelem,ivar) + dudy->get(jelem,ivar)*dudy->get(jelem,ivar) + epsilon, gamma);
					w = 1.0 / denom;
					wsum += w;
					(*ldudx)(ielem,ivar) += w*dudx->get(jelem,ivar);
					(*ldudy)(ielem,ivar) += w*dudy->get(jelem,ivar);
				}

				(*ldudx)(ielem,ivar) /= wsum;
				(*ldudy)(ielem,ivar) /= wsum;
			}
		}
		
		// internal faces
#pragma omp for
		for(acfd_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			acfd_int ielem = m->gintfac(ied,0);
			acfd_int jelem = m->gintfac(ied,1);

			//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
			for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
			{
				for(int i = 0; i < NVARS; i++)
				{

					(*ufl)(ied,i) = u->get(ielem,i) + ldudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + ldudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
					(*ufr)(ied,i) = u->get(jelem,i) + ldudx->get(jelem,i)*(gr[ied].get(ig,0)-ri->get(jelem,0)) + ldudy->get(jelem,i)*(gr[ied].get(ig,1)-ri->get(jelem,1));
				}
			}
		}
		
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
#pragma omp for
		for(acfd_int ied = 0; ied < m->gnbface(); ied++)
		{
			acfd_int ielem = m->gintfac(ied,0);

			for(int ig = 0; ig < ng; ig++)
			{
				for(int i = 0; i < NVARS; i++)
					(*ufl)(ied,i) = u->get(ielem,i) + ldudx->get(ielem,i)*(gr[ied].get(ig,0)-ri->get(ielem,0)) + ldudy->get(ielem,i)*(gr[ied].get(ig,1)-ri->get(ielem,1));
			}
		}
	} // end parallel region
}

void VanAlbadaLimiter::setup(const UMesh2dh* mesh, const amat::Matrix<acfd_real>* unknowns, const amat::Matrix<acfd_real>* unknow_ghost, 
	const amat::Matrix<acfd_real>* x_deriv, const amat::Matrix<acfd_real>* y_deriv, 
	const amat::Matrix<acfd_real>* ghost_centres, const amat::Matrix<acfd_real>* r_centres, 
	const amat::Matrix<acfd_real>* gauss_r, amat::Matrix<acfd_real>* uface_left, amat::Matrix<acfd_real>* uface_right)
{
	FaceDataComputation::setup(mesh, unknowns, unknow_ghost, x_deriv, y_deriv, ghost_centres, r_centres, gauss_r, uface_left, uface_right);
	eps = 1e-8;
	k = 1.0/3.0;
	phi_l.setup(m->gnaface(), NVARS);
	phi_r.setup(m->gnaface(), NVARS);
}

/// Calculate values of variables at left and right sides of each face based on computed derivatives and limiter values
void VanAlbadaLimiter::compute_face_values()
{
	//compute_limiters
	
	for(acfd_int ied = 0; ied < m->gnbface(); ied++)
	{
		int lel = m->gintfac(ied,0);
		amat::Matrix<acfd_real> deltam(NVARS,1);
		for(int i = 0; i < NVARS; i++)
		{
			deltam(i) = 2 * ( dudx->get(lel,i)*(rb->get(ied,0)-ri->get(lel,0)) + dudy->get(lel,i)*(rb->get(ied,1)-ri->get(lel,1)) ) - (ug->get(ied,i) - u->get(lel,i));
			phi_l(ied,i) = (2*deltam(i) * (ug->get(ied,i) - u->get(lel,i)) + eps) / (deltam(i)*deltam(i) + (ug->get(ied,i) - u->get(lel,i))*(ug->get(ied,i) - u->get(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;
		}
	}

	for(acfd_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		acfd_int lel = m->gintfac(ied,0);
		acfd_int rel = m->gintfac(ied,1);
		amat::Matrix<acfd_real> deltam(NVARS,1);
		amat::Matrix<acfd_real> deltap(NVARS,1);
		for(int i = 0; i < NVARS; i++)
		{
			deltam(i) = 2 * ( dudx->get(lel,i)*(ri->get(rel,0)-ri->get(lel,0)) + dudy->get(lel,i)*(ri->get(rel,1)-ri->get(lel,1)) ) - (u->get(rel,i) - u->get(lel,i));
			deltap(i) = 2 * ( dudx->get(rel,i)*(ri->get(rel,0)-ri->get(lel,0)) + dudy->get(rel,i)*(ri->get(rel,1)-ri->get(lel,1)) ) - (u->get(rel,i) - u->get(lel,i));

			phi_l(ied,i) = (2*deltam(i) * (u->get(rel,i) - u->get(lel,i)) + eps) / (deltam(i)*deltam(i) + (u->get(rel,i) - u->get(lel,i))*(u->get(rel,i) - u->get(lel,i)) + eps);
			if( phi_l(ied,i) < 0.0) phi_l(ied,i) = 0.0;

			phi_r(ied,i) = (2*deltap(i) * (u->get(rel,i) - u->get(lel,i)) + eps) / (deltap(i)*deltap(i) + (u->get(rel,i) - u->get(lel,i))*(u->get(rel,i) - u->get(lel,i)) + eps);
			if( phi_r(ied,i) < 0.0) phi_r(ied,i) = 0.0;
		}
	}

	// apply the limiters
	
	//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - internal\n";
	amat::Matrix<acfd_real> deltam(NVARS,1);
	amat::Matrix<acfd_real> deltap(NVARS,1);
	for(acfd_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		acfd_int ielem = m->gintfac(ied,0);
		acfd_int jelem = m->gintfac(ied,1);

		// NOTE: Only for 1 Gauss point per face
		//cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
		for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
		{
			for(int i = 0; i < NVARS; i++)
			{
				deltam(i) = 2 * ( dudx->get(ielem,i)*(ri->get(jelem,0)-ri->get(ielem,0)) + dudy->get(ielem,i)*(ri->get(jelem,1)-ri->get(ielem,1)) ) - (u->get(jelem,i) - u->get(ielem,i));
				deltap(i) = 2 * ( dudx->get(jelem,i)*(ri->get(jelem,0)-ri->get(ielem,0)) + dudy->get(jelem,i)*(ri->get(jelem,1)-ri->get(ielem,1)) ) - (u->get(jelem,i) - u->get(ielem,i));

				(*ufl)(ied,i) = u->get(ielem,i) + phi_l.get(ied,i)/4.0*( (1-k*phi_l.get(ied,i))*deltam.get(i) + (1+k*phi_l.get(ied,i))*(u->get(jelem,i) - u->get(ielem,i)) );
				(*ufr)(ied,i) = u->get(jelem,i) + phi_r.get(ied,i)/4.0*( (1-k*phi_r.get(ied,i))*deltap(i) + (1+k*phi_r.get(ied,i))*(u->get(jelem,i) - u->get(ielem,i)) );
			}
		}
	}
}

/*void VanAlbadaLimiter::compute_reg_face_values()
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
			for(int i = 0; i < NVARS; i++)
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
			for(int i = 0; i < NVARS; i++)
				(*ufl)(ied,i) = u->get(ielem,i) + phi_l.get(ied,i)*dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + phi_l.get(ied,i)*dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));

		}
	}
}*/

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
    int ngauss;

    Matrix<double>* phi_l;
    Matrix<double>* phi_r;

    Matrix<double> umax;        // holds, for each element, max value of u over all its face-neighbors
    Matrix<double> umin;        // holds, for each element, min value of u over all its face-neighbors

    void compute_maxmin()
    {
        for(int iel = 0; iel < m->gnelem(); iel++)
        {
            for(int i = 0; i < NVARS; i++)
            {
                umax(iel,i) = u->get(m->gesuel(iel,0),i);
                umin(iel,i) = u->get(m->gesuel(iel,0),i);
            }

            for(int iface = 1; iface < m->gnfael(iel); iface++)
            {
                for(int i = 0; i < NVARS; i++)
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
        dudx = x_deriv;
        dudy = y_deriv;
        xi = x_centres;
        yi = y_centres;
        gaussx = x_gauss;
        gaussy = y_gauss;
        phi_l = limits_l;
        phi_r = limits_r;

        ngauss = gaussx->cols();

        umax.setup(m->gnelem(), NVARS, ROWMAJOR);
        umin.setup(m->gnelem(), NVARS, ROWMAJOR);
    }

    void compute_limiters()
    {
        compute_maxmin();

        //Matrix<double>* phig;       // one matrix for each element - each matrix stores phi_{i,g} for each gauss point g and for all 4 variables
        //phig = new Matrix<double>[m->gnelem()];
        //for(int iel = 0; iel < m->gnelem(); iel++)
        //	phig[iel].setup(ngauss,NVARS,ROWMAJOR);

        for(int ied = 0; ied < m->gnbface(); ied++)
        {
            int lel = m->gintfac(ied,0);
            Matrix<double> ugminus(ngauss,NVARS,ROWMAJOR);
            Matrix<double> ugplus(ngauss, NVARS, ROWMAJOR);
            Matrix<double> phig_l(ngauss,NVARS,ROWMAJOR);
            Matrix<double> phig_r(ngauss,NVARS,ROWMAJOR);
            for(int g = 0; g < ngauss; g++)
            {
                for(int i = 0; i < NVARS; i++)
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
