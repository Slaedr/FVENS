#ifdef _OPENMP
#ifndef OMP_H
#include <omp.h>
#endif
#endif

#ifndef _GLIBCXX_CSTDIO
#include <cstdio>
#endif

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#define __ALIMITER_H

#define NTHREADS 4

using namespace std;
using namespace amat;
using namespace acfd;

namespace acfd {

class BarthJaspersonLimiter
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

        /*Matrix<double>* phig;       // one matrix for each element - each matrix stores phi_{i,g} for each gauss point g and for all 4 variables
        phig = new Matrix<double>[m->gnelem()];
        for(int iel = 0; iel < m->gnelem(); iel++)
            phig[iel].setup(ngauss,nvars,ROWMAJOR);*/

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
};

class VanAlbadaLimiter
{
    UTriMesh* m;
    Matrix<double>* u;
    Matrix<double>* ug;
    Matrix<double>* uinf;
    Matrix<double>* dudx;
    Matrix<double>* dudy;
    Matrix<double>* xb;
    Matrix<double>* yb;
    Matrix<double>* xi;
    Matrix<double>* yi;
    Matrix<double>* gx;     // x-coords of gauss points of each cell - needed to calculate interface values at these points
    Matrix<double>* gy;
    int nvars;
    int ng;                 // number of gauss points
    double eps;
    double k;               // van-Albada parameter
    double g;               // adiabatic index

    Matrix<double>* phi_l;
    Matrix<double>* phi_r;
    Matrix<double>* ufl;
    Matrix<double>* ufr;

public:
    VanAlbadaLimiter() {}

    void setup_limiter(UTriMesh* mesh, Matrix<double>* unknowns, Matrix<double>* unknow_ghost, Matrix<double>* x_deriv, Matrix<double>* y_deriv, Matrix<double>* x_ghost_centres, Matrix<double>* y_ghost_centres, Matrix<double>* x_centres, Matrix<double>* y_centres, Matrix<double>* gauss_x, Matrix<double>* gauss_y, Matrix<double>* limits_left, Matrix<double>* limits_right, Matrix<double>* uface_left, Matrix<double>* uface_right, Matrix<double>* u_farfield, double adiabatic_index)
    {
        m = mesh;
        u = unknowns;
        ug = unknow_ghost;          // contains ghost cell states according to BCs for each boundary edge
        uinf = u_farfield;
        nvars = u->cols();
        eps = 1e-10;
        k = 1.0/3.0;
        dudx = x_deriv;
        dudy = y_deriv;
        xi = x_centres;
        yi = y_centres;
        xb = x_ghost_centres;       // contains x-coord of right "cell centroid" of each boundary edge
        yb = y_ghost_centres;       // contains y-coord of right "cell centroid" of each boundary edge
        gx = gauss_x;
        gy = gauss_y;
        phi_l = limits_left;
        phi_r = limits_right;
        ufl = uface_left;
        ufr = uface_right;
        ng = gx->cols();
        g = adiabatic_index;
    }

    void compute_limiters()
    {
		Matrix<double> deltam(nvars,1,ROWMAJOR);
        Matrix<double> deltap(nvars,1,ROWMAJOR);
        int lel, rel, ied;
        for(ied = 0; ied < m->gnbface(); ied++)
        {
            lel = m->gintfac(ied,0);
            for(int i = 0; i < nvars; i++)
            {
                deltam(i) = 2 * ( dudx->get(lel,i)*(xb->get(ied)-xi->get(lel)) + dudy->get(lel,i)*(yb->get(ied)-yi->get(lel)) ) - (ug->get(ied,i) - u->get(lel,i));
                (*phi_l)(ied,i) = (2*deltam(i) * (ug->get(ied,i) - u->get(lel,i)) + eps) / (deltam(i)*deltam(i) + (ug->get(ied,i) - u->get(lel,i))*(ug->get(ied,i) - u->get(lel,i)) + eps);
                if( (*phi_l)(ied,i) < 0.0) (*phi_l)(ied,i) = 0.0;
            }
        }
		
        for(ied = m->gnbface(); ied < m->gnaface(); ied++)
        {
            lel = m->gintfac(ied,0);
            rel = m->gintfac(ied,1);
            
            for(int i = 0; i < nvars; i++)
            {
                deltam(i) = 2 * ( dudx->get(lel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(lel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));
                deltap(i) = 2 * ( dudx->get(rel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(rel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));

                (*phi_l)(ied,i) = (2*deltam(i) * (u->get(rel,i) - u->get(lel,i)) + eps) / (deltam(i)*deltam(i) + (u->get(rel,i) - u->get(lel,i))*(u->get(rel,i) - u->get(lel,i)) + eps);
                if( (*phi_l)(ied,i) < 0.0) (*phi_l)(ied,i) = 0.0;

                (*phi_r)(ied,i) = (2*deltap(i) * (u->get(rel,i) - u->get(lel,i)) + eps) / (deltap(i)*deltap(i) + (u->get(rel,i) - u->get(lel,i))*(u->get(rel,i) - u->get(lel,i)) + eps);
                if( (*phi_r)(ied,i) < 0.0) (*phi_r)(ied,i) = 0.0;
            }
        }
    }

    void compute_interface_values()
    {
        // Calculate values of variables at left and right sides of each face based on computed derivatives
        // if parallel, copy class variables into local variables
        #ifdef _OPENMP
        UTriMesh* m = VanAlbadaLimiter::m;
        Matrix<double>* u = VanAlbadaLimiter::u;
        Matrix<double>* ug = VanAlbadaLimiter::ug;
        Matrix<double>* uinf = VanAlbadaLimiter::uinf;
        Matrix<double>* ufl = VanAlbadaLimiter::ufl;
        Matrix<double>* ufr = VanAlbadaLimiter::ufr;
        Matrix<double>* phi_l = VanAlbadaLimiter::phi_l;
        Matrix<double>* phi_r = VanAlbadaLimiter::phi_r;
        Matrix<double>* dudx = VanAlbadaLimiter::dudx;
        Matrix<double>* dudy = VanAlbadaLimiter::dudy;
        Matrix<double>* xi = VanAlbadaLimiter::xi;
        Matrix<double>* yi = VanAlbadaLimiter::yi;
        Matrix<double>* xb = VanAlbadaLimiter::xb;
        Matrix<double>* yb = VanAlbadaLimiter::yb;
        int nvars = VanAlbadaLimiter::nvars;
        int ng = VanAlbadaLimiter::ng;
        #endif
        
        int ied, lel, rel;
        
        // (a) internal faces
        //cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - internal\n";
        #pragma omp parallel for default(none) private(ied,lel,rel) shared(m,ufl,ufr,phi_l,phi_r,dudx,dudy,u,xi,yi,nvars,ng) num_threads(NTHREADS)
		for(ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			Matrix<double> deltam(nvars,1,ROWMAJOR);
			Matrix<double> deltap(nvars,1,ROWMAJOR);
        
			lel = m->gintfac(ied,0);
			rel = m->gintfac(ied,1);

            // TODO: correct for multiple gauss points
            //cout << "VanAlbadaLimiter: compute_interface_values(): iterate over gauss points..\n";
            for(int ig = 0; ig < ng; ig++)      // iterate over gauss points
            {
    			for(int i = 0; i < nvars; i++)
                {
                    //printf("Setting deltas\n");
                    deltam(i) = 2 * ( dudx->get(lel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(lel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));
                    deltap(i) = 2 * ( dudx->get(rel,i)*(xi->get(rel)-xi->get(lel)) + dudy->get(rel,i)*(yi->get(rel)-yi->get(lel)) ) - (u->get(rel,i) - u->get(lel,i));
					//printf("Setting ufl and ufr\n");
                    //ufl(ied,i) = u->get(ielem,i) + phi_l(ied,i)*dudx->get(ielem,i)*(gaussx(ied,0)-xi(ielem)) + phi_l(ied,i)*dudy->get(ielem,i)*(gaussy(ied,0)-yi(ielem));
    				//ufr(ied,i) = u->get(jelem,i) + phi_r(ied,i)*dudx->get(jelem,i)*(gaussx(ied,0)-xi(jelem)) + phi_r(ied,i)*dudy->get(jelem,i)*(gaussy(ied,0)-yi(jelem));
    				(*ufl)(ied,i) = u->get(lel,i) + phi_l->get(ied,i)/4.0*( (1-k*phi_l->get(ied,i))*deltam.get(i) + (1+k*phi_l->get(ied,i))*(u->get(rel,i) - u->get(lel,i)) );
    				(*ufr)(ied,i) = u->get(rel,i) - phi_r->get(ied,i)/4.0*( (1-k*phi_r->get(ied,i))*deltap(i) + (1+k*phi_r->get(ied,i))*(u->get(rel,i) - u->get(lel,i)) );
    			}
            }
		}
		
		//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - boundary\n";
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
		// (b) boundary faces
        double nx, ny, vni, pi, ci;
        
        #pragma omp parallel for default(none) private(ied,lel,pi,ci,vni,nx,ny) shared(m,ufl,ufr,phi_l,dudx,dudy,u,ug,uinf,xi,yi,xb,yb,nvars) num_threads(NTHREADS)
		for(ied = 0; ied < m->gnbface(); ied++)
		{
			Matrix<double> deltam(nvars,1,ROWMAJOR);
			
			lel = m->gintfac(ied,0);
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);

			for(int i = 0; i < nvars; i++)
			{
				deltam(i) = 2 * ( dudx->get(lel,i)*(xb->get(ied)-xi->get(lel)) + dudy->get(lel,i)*(yb->get(ied)-yi->get(lel)) ) - (ug->get(ied,i) - u->get(lel,i));

				(*ufl)(ied,i) = u->get(lel,i) + phi_l->get(ied,i)/4.0*( (1-k*phi_l->get(ied,i))*deltam(i) + (1+k*phi_l->get(ied,i))*(ug->get(ied,i) - u->get(lel,i)) );
			}

			vni = (ufl->get(ied,1)*nx + ufl->get(ied,2)*ny)/ufl->get(ied,0);
			pi = (g-1)*(ufl->get(ied,3) - 0.5*(pow(ufl->get(ied,1),2)+pow(ufl->get(ied,2),2))/ufl->get(ied,0));
			ci = sqrt(g*pi/ufl->get(ied,0));

			if(m->ggallfa(ied,3) == 2)		// solid wall
			{
				(*ufr)(ied,0) = ufl->get(ied,0);
				(*ufr)(ied,1) = ufl->get(ied,1) - 2*vni*nx*ufr->get(ied,0);
				(*ufr)(ied,2) = ufl->get(ied,2) - 2*vni*ny*ufr->get(ied,0);
				(*ufr)(ied,3) = ufl->get(ied,3);
			}
			if(m->ggallfa(ied,3) == 4)		// inflow or outflow
			{
				/*if(Mni < -1.0)
				{
					for(int i = 0; i < nvars; i++)
						ufr(ied,i) = uinf->get(0,i);
				}
				else if(Mni >= -1.0 && Mni < 0.0)
				{
					double vinfx = uinf->get(0,1)/uinf->get(0,0);
					double vinfy = uinf->get(0,2)/uinf->get(0,0);
					double vinfn = vinfx*nx + vinfy*ny;
					double vbn = (ufl->get(lel,1)*nx + ufl->get(lel,2)*ny)/ufl->get(lel,0);
					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
					double pb = (g-1)*(ufl->get(lel,3) - 0.5*(pow(ufl->get(lel,1),2)+pow(ufl->get(lel,2),2))/ufl->get(lel,0));
					double cinf = sqrt(g*pinf/uinf->get(0,0));
					double cb = sqrt(g*pb/ufl->get(lel,0));

					double vgx = vinfx*ny*ny - vinfy*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
					double vgy = vinfy*nx*nx - vinfx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
					vnj = vgx*nx + vgy*ny;	// = vgn
					cj = (g-1)/2*(vnj-vinfn)+cinf;
					ufr(ied,0) = pow( pinf/pow(uinf->get(0,0),g) * 1.0/cj*cj , 1/(1-g));	// density
					pj = ufr(ied,0)/g*cj*cj;

					ufr(ied,3) = pj/(g-1) + 0.5*ufr(ied,0)*(vgx*vgx+vgy*vgy);
					ufr(ied,1) = ufr(ied,0)*vgx;
					ufr(ied,2) = ufr(ied,0)*vgy;
				}
				else if(Mni >= 0.0 && Mni < 1.0)
				{
					double vbx = ufl(lel,1)/ufl(lel,0);
					double vby = ufl(lel,2)/ufl(lel,0);
					double vbn = vbx*nx + vby*ny;
					double vinfn = uinf->get(0,1)/uinf->get(0,0)*nx + uinf->get(0,2)/uinf->get(0,0)*ny;
					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
					double pb = (g-1)*(ufl(lel,3) - 0.5*(pow(ufl(lel,1),2)+pow(ufl(lel,2),2))/ufl(lel,0));
					double cinf = sqrt(g*pinf/uinf->get(0,0));
					double cb = sqrt(g*pb/ufl(lel,0));

					double vgx = vbx*ny*ny - vby*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
					double vgy = vby*nx*nx - vbx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
					vnj = vgx*nx + vgy*ny;	// = vgn
					cj = (g-1)/2*(vnj-vinfn)+cinf;
					ufr(ied,0) = pow( pb/pow(ufl(lel,0),g) * 1.0/cj*cj , 1/(1-g));	// density
					pj = ufr(ied,0)/g*cj*cj;

					ufr(ied,3) = pj/(g-1) + 0.5*ufr(ied,0)*(vgx*vgx+vgy*vgy);
					ufr(ied,1) = ufr(ied,0)*vgx;
					ufr(ied,2) = ufr(ied,0)*vgy;
				}
				else
				{
					for(int i = 0; i < nvars; i++)
						ufr(ied,i) = ufl(lel,i);
				} */

				// Naive way
				for(int i = 0; i < nvars; i++)
					(*ufr)(ied,i) = uinf->get(0,i);
			}
		}
    }

    void compute_unlimited_interface_values()
    {
        #ifdef _OPENMP
        UTriMesh* m = VanAlbadaLimiter::m;
        Matrix<double>* u = VanAlbadaLimiter::u;
        //Matrix<double>* ug = VanAlbadaLimiter::ug;
        Matrix<double>* uinf = VanAlbadaLimiter::uinf;
        Matrix<double>* ufl = VanAlbadaLimiter::ufl;
        Matrix<double>* ufr = VanAlbadaLimiter::ufr;
        Matrix<double>* dudx = VanAlbadaLimiter::dudx;
        Matrix<double>* dudy = VanAlbadaLimiter::dudy;
        Matrix<double>* xi = VanAlbadaLimiter::xi;
        Matrix<double>* yi = VanAlbadaLimiter::yi;
        Matrix<double>* xb = VanAlbadaLimiter::xb;
        Matrix<double>* yb = VanAlbadaLimiter::yb;
        Matrix<double>* gx = VanAlbadaLimiter::gx;
        Matrix<double>* gy = VanAlbadaLimiter::gy;
        int nvars = VanAlbadaLimiter::nvars;
        int ng = VanAlbadaLimiter::ng;
        #endif
        
        int ied, lel, rel, ielem, jelem;
        
        // Calculate values of variables at left and right sides of each face based on computed derivatives
		// (a) internal faces
        //cout << "VanAlbadaLimiter: compute_unlimited_interface_values(): Computing values at faces - internal\n";
        #pragma omp parallel for default(none) private(ied,ielem,jelem) shared(m,ufl,ufr,dudx,dudy,u,xi,yi,gx,gy,nvars,ng) num_threads(NTHREADS)
		for(ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			ielem = m->gintfac(ied,0);
			jelem = m->gintfac(ied,1);

            // TODO: correct for multiple gauss points
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
		//cout << "VanAlbadaLimiter: compute_unlimited_interface_values(): Computing values at faces - boundary\n";
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
		// (b) boundary faces
		double vni, pi, ci, nx, ny;
		
		#pragma omp parallel for default(none) private(ied,ielem,vni,pi,ci,nx,ny) shared(m,ufl,ufr,dudx,dudy,u,uinf,xi,yi,gx,gy,nvars,ng) num_threads(NTHREADS)
		for(ied = 0; ied < m->gnbface(); ied++)
		{
			ielem = m->gintfac(ied,0); int lel = ielem;
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);

            for(int ig = 0; ig < ng; ig++)
            {
                for(int i = 0; i < nvars; i++)
                {

                    (*ufl)(ied,i) = u->get(ielem,i) + dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));
    			}

    			vni = (ufl->get(ied,1)*nx + ufl->get(ied,2)*ny);
    			//pi = (g-1)*(ufl->get(ied,3) - 0.5*(pow(ufl->get(ied,1),2)+pow(ufl->get(ied,2),2))/ufl->get(ied,0));
    			//ci = sqrt(g*pi/ufl->get(ied,0));
    			if(m->ggallfa(ied,3) == 2)		// solid wall
    			{
    				(*ufr)(ied,0) = ufl->get(ied,0);
    				(*ufr)(ied,1) = ufl->get(ied,1) - 2*vni*nx;
    				(*ufr)(ied,2) = ufl->get(ied,2) - 2*vni*ny;
    				(*ufr)(ied,3) = ufl->get(ied,3);
    			}
    			if(m->ggallfa(ied,3) == 4)		// inflow or outflow
    			{
    				/*if(Mni < -1.0)        // characteristic boundary conditions
    				{
    					for(int i = 0; i < nvars; i++)
    						ufr(ied,i) = uinf->get(0,i);
    				}
    				else if(Mni >= -1.0 && Mni < 0.0)
    				{
    					double vinfx = uinf->get(0,1)/uinf->get(0,0);
    					double vinfy = uinf->get(0,2)/uinf->get(0,0);
    					double vinfn = vinfx*nx + vinfy*ny;
    					double vbn = ufl(lel,1)/ufl(lel,0)*nx + ufl(lel,2)/ufl(lel,0)*ny;
    					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
    					double pb = (g-1)*(ufl(lel,3) - 0.5*(pow(ufl(lel,1),2)+pow(ufl(lel,2),2))/ufl(lel,0));
    					double cinf = sqrt(g*pinf/uinf->get(0,0));
    					double cb = sqrt(g*pb/ufl(lel,0));

    					double vgx = vinfx*ny*ny - vinfy*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
    					double vgy = vinfy*nx*nx - vinfx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
    					vnj = vgx*nx + vgy*ny;	// = vgn
    					cj = (g-1)/2*(vnj-vinfn)+cinf;
    					ufr(ied,0) = pow( pinf/pow(uinf->get(0,0),g) * 1.0/cj*cj , 1/(1-g));	// density
    					pj = ufr(ied,0)/g*cj*cj;

    					ufr(ied,3) = pj/(g-1) + 0.5*ufr(ied,0)*(vgx*vgx+vgy*vgy);
    					ufr(ied,1) = ufr(ied,0)*vgx;
    					ufr(ied,2) = ufr(ied,0)*vgy;
    				}
    				else if(Mni >= 0.0 && Mni < 1.0)
    				{
    					double vbx = ufl(lel,1)/ufl(lel,0);
    					double vby = ufl(lel,2)/ufl(lel,0);
    					double vbn = vbx*nx + vby*ny;
    					double vinfn = uinf->get(0,1)/uinf->get(0,0)*nx + uinf->get(0,2)/uinf->get(0,0)*ny;
    					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
    					double pb = (g-1)*(ufl(lel,3) - 0.5*(pow(ufl(lel,1),2)+pow(ufl(lel,2),2))/ufl(lel,0));
    					double cinf = sqrt(g*pinf/uinf->get(0,0));
    					double cb = sqrt(g*pb/ufl(lel,0));

    					double vgx = vbx*ny*ny - vby*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
    					double vgy = vby*nx*nx - vbx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
    					vnj = vgx*nx + vgy*ny;	// = vgn
    					cj = (g-1)/2*(vnj-vinfn)+cinf;
    					ufr(ied,0) = pow( pb/pow(ufl(lel,0),g) * 1.0/cj*cj , 1/(1-g));	// density
    					pj = ufr(ied,0)/g*cj*cj;

    					ufr(ied,3) = pj/(g-1) + 0.5*ufr(ied,0)*(vgx*vgx+vgy*vgy);
    					ufr(ied,1) = ufr(ied,0)*vgx;
    					ufr(ied,2) = ufr(ied,0)*vgy;
    				}
    				else
    				{
    					for(int i = 0; i < nvars; i++)
    						ufr(ied,i) = ufl(lel,i);
    				} */

    				// Naive way
    				for(int i = 0; i < nvars; i++)
    					(*ufr)(ied,i) = uinf->get(0,i);
    			}
            }
		}
    }
};

} // end namespace
