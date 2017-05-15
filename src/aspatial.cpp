/** @file aspatial.cpp
 * @brief Finite volume spatial discretization of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 */

#include "aspatial.hpp"

namespace acfd {

EulerFV::EulerFV(const UMesh2dh* mesh, const int _order, std::string invflux, std::string jacflux, std::string reconst, std::string limiter) 
	: aflux(1.4)
{
	m = mesh;
	order = _order;
	g = 1.4;

	std::cout << "EulerFV: Setting up explicit solver for spatial order " << order << std::endl;

	// for upto second-order finite volume, we only need 1 Guass point per face
	ngaussf = 1;

	/// TODO: Take the two values below as input from control file, rather than hardcoding
	solid_wall_id = 2;
	inflow_outflow_id = 4;

	// allocation
	residual.setup(m->gnelem(),NVARS);
	u.setup(m->gnelem(), NVARS);
	uinf.setup(1, NVARS);
	integ.setup(m->gnelem(), 1);
	dtm.setup(m->gnelem(), 1);
	dudx.setup(m->gnelem(), NVARS);
	dudy.setup(m->gnelem(), NVARS);
	fluxes.setup(m->gnaface(), NVARS);
	uleft.setup(m->gnaface(), NVARS);
	uright.setup(m->gnaface(), NVARS);
	rc.setup(m->gnelem(),m->gndim());
	rcg.setup(m->gnface(),m->gndim());
	ug.setup(m->gnface(),NVARS);
	gr = new amat::Matrix<a_real>[m->gnaface()];
	for(int i = 0; i <  m->gnaface(); i++)
		gr[i].setup(ngaussf, m->gndim());

	// set inviscid flux scheme
	if(invflux == "VANLEER")
		inviflux = new VanLeerFlux(g, &aflux);
	else if(invflux == "ROE")
	{
		inviflux = new RoeFlux(g, &aflux);
		std::cout << "EulerFV: Using Roe fluxes." << std::endl;
	}
	else if(invflux == "HLLC")
	{
		inviflux = new HLLCFlux(g, &aflux);
		std::cout << "EulerFV: Using HLLC fluxes." << std::endl;
	}
	else if(invflux == "LLF")
	{
		inviflux = new LocalLaxFriedrichsFlux(g, &aflux);
		std::cout << "EulerFV: Using LLF fluxes." << std::endl;
	}
	else
		std::cout << "EulerFV: ! Flux scheme not available!" << std::endl;
	
	// set inviscid flux scheme for Jacobian
	if(invflux == "VANLEER")
		inviflux = new VanLeerFlux(g, &aflux);
	else if(invflux == "ROE")
	{
		inviflux = new RoeFlux(g, &aflux);
		std::cout << "EulerFV: Using Roe fluxes." << std::endl;
	}
	else if(invflux == "HLLC")
	{
		inviflux = new HLLCFlux(g, &aflux);
		std::cout << "EulerFV: Using HLLC fluxes." << std::endl;
	}
	else if(invflux == "LLF")
	{
		inviflux = new LocalLaxFriedrichsFlux(g, &aflux);
		std::cout << "EulerFV: Using LLF fluxes for Jacobian." << std::endl;
	}
	else
		std::cout << "EulerFV: ! Flux scheme not available!" << std::endl;

	// set reconstruction scheme
	std::cout << "EulerFV: Selected reconstruction scheme is " << reconst << std::endl;
	if(reconst == "GREENGAUSS")
	{
		rec = new GreenGaussReconstruction();
	}
	else 
	{
		rec = new WeightedLeastSquaresReconstruction();
	}
	if(order == 1) std::cout << "EulerFV: No reconstruction will be used." << std::endl;

	// set limiter
	if(limiter == "NONE")
	{
		lim = new NoLimiter(m, &u, &ug, &dudx, &dudy, &rcg, &rc, gr, &uleft, &uright);
		std::cout << "EulerFV: No limiter will be used." << std::endl;
	}
	else if(limiter == "WENO")
	{
		lim = new WENOLimiter(m, &u, &ug, &dudx, &dudy, &rcg, &rc, gr, &uleft, &uright);
		std::cout << "EulerFV: WENO limiter selected.\n";
	}
}

EulerFV::~EulerFV()
{
	delete rec;
	delete inviflux;
	delete lim;
	delete [] gr;
}

void EulerFV::compute_ghost_cell_coords_about_midpoint()
{
	int iface, ielem, idim, ip1, ip2;
	std::vector<a_real> midpoint(m->gndim());
	for(iface = 0; iface < m->gnbface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		ip1 = m->gintfac(iface,2);
		ip2 = m->gintfac(iface,3);

		for(idim = 0; idim < m->gndim(); idim++)
		{
			midpoint[idim] = 0.5 * (m->gcoords(ip1,idim) + m->gcoords(ip2,idim));
		}

		for(idim = 0; idim < m->gndim(); idim++)
			rcg(iface,idim) = 2*midpoint[idim] - rc(ielem,idim);
	}
}

void EulerFV::compute_ghost_cell_coords_about_face()
{
	int ied, ielem;
	a_real x1, y1, x2, y2, xs, ys, xi, yi;

	for(ied = 0; ied < m->gnbface(); ied++)
	{
		ielem = m->gintfac(ied,0); //int lel = ielem;
		//jelem = m->gintfac(ied,1); //int rel = jelem;
		a_real nx = m->ggallfa(ied,0);
		a_real ny = m->ggallfa(ied,1);

		xi = rc.get(ielem,0);
		yi = rc.get(ielem,1);

		// Note: The ghost cell is a direct reflection of the boundary cell about the boundary-face
		//       It is NOT the reflection about the midpoint of the boundary-face
		x1 = m->gcoords(m->gintfac(ied,2),0);
		x2 = m->gcoords(m->gintfac(ied,3),0);
		y1 = m->gcoords(m->gintfac(ied,2),1);
		y2 = m->gcoords(m->gintfac(ied,3),1);

		if(fabs(nx)>A_SMALL_NUMBER && fabs(ny)>A_SMALL_NUMBER)		// check if nx != 0 and ny != 0
		{
			xs = ( yi-y1 - ny/nx*xi + (y2-y1)/(x2-x1)*x1 ) / ((y2-y1)/(x2-x1)-ny/nx);
			ys = ny/nx*xs + yi - ny/nx*xi;
		}
		else if(fabs(nx)<=A_SMALL_NUMBER)
		{
			xs = xi;
			ys = y1;
		}
		else
		{
			xs = x1;
			ys = yi;
		}
		rcg(ied,0) = 2*xs-xi;
		rcg(ied,1) = 2*ys-yi;
	}
}

/// Function to feed needed data, and compute cell-centers
/** \param Minf Free-stream Mach number
 * \param vinf Free stream velocity magnitude
 * \param a Angle of attack (radians)
 * \param rhoinf Free stream density
 */
void EulerFV::loaddata(a_real Minf, a_real vinf, a_real a, a_real rhoinf)
{
	// Note that reference density and reference velocity are the values at infinity
	//std::cout << "EulerFV: loaddata(): Calculating initial data...\n";
	a_real vx = vinf*cos(a);
	a_real vy = vinf*sin(a);
	a_real p = rhoinf*vinf*vinf/(g*Minf*Minf);
	uinf(0,0) = rhoinf;		// should be 1
	uinf(0,1) = rhoinf*vx;
	uinf(0,2) = rhoinf*vy;
	uinf(0,3) = p/(g-1) + 0.5*rhoinf*vinf*vinf;

	//initial values are equal to boundary values
	for(int i = 0; i < m->gnelem(); i++)
		for(int j = 0; j < NVARS; j++)
			u(i,j) = uinf(0,j);

	// Next, get cell centers (real and ghost)
	
	int idim, inode;

	for(int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(idim = 0; idim < m->gndim(); idim++)
		{
			rc(ielem,idim) = 0;
			for(inode = 0; inode < m->gnnode(ielem); inode++)
				rc(ielem,idim) += m->gcoords(m->ginpoel(ielem, inode), idim);
			rc(ielem,idim) = rc(ielem,idim) / (a_real)(m->gnnode(ielem));
		}
	}

	int ied, ig;
	a_real x1, y1, x2, y2;

	compute_ghost_cell_coords_about_midpoint();
	//compute_ghost_cell_coords_about_face();

	//Calculate and store coordinates of Gauss points (general implementation)
	// Gauss points are uniformly distributed along the face.
	for(ied = 0; ied < m->gnaface(); ied++)
	{
		x1 = m->gcoords(m->gintfac(ied,2),0);
		y1 = m->gcoords(m->gintfac(ied,2),1);
		x2 = m->gcoords(m->gintfac(ied,3),0);
		y2 = m->gcoords(m->gintfac(ied,3),1);
		for(ig = 0; ig < ngaussf; ig++)
		{
			gr[ied](ig,0) = x1 + (a_real)(ig+1.0)/(a_real)(ngaussf+1.0) * (x2-x1);
			gr[ied](ig,1) = y1 + (a_real)(ig+1.0)/(a_real)(ngaussf+1.0) * (y2-y1);
		}
	}

	rec->setup(m, &u, &ug, &dudx, &dudy, &rc, &rcg);
	std::cout << "EulerFV: loaddata(): Initial data calculated.\n";
}

void EulerFV::compute_boundary_states(const amat::Matrix<a_real>& ins, amat::Matrix<a_real>& bs)
{
#pragma omp parallel for default(shared)
	for(int ied = 0; ied < m->gnbface(); ied++)
	{
		a_real nx = m->ggallfa(ied,0);
		a_real ny = m->ggallfa(ied,1);

		a_real vni = (ins.get(ied,1)*nx + ins.get(ied,2)*ny)/ins.get(ied,0);
		//a_real pi = (g-1.0)*(ins.get(ied,3) - 0.5*(pow(ins.get(ied,1),2)+pow(ins.get(ied,2),2))/ins.get(ied,0));
		//a_real pinf = (g-1.0)*(uinf.get(0,3) - 0.5*(pow(uinf.get(0,1),2)+pow(uinf.get(0,2),2))/uinf.get(0,0));
		//a_real ci = sqrt(g*pi/ins.get(ied,0));
		//a_real Mni = vni/ci;

		if(m->ggallfa(ied,3) == solid_wall_id)
		{
			bs(ied,0) = ins.get(ied,0);
			bs(ied,1) = ins.get(ied,1) - 2*vni*nx*bs(ied,0);
			bs(ied,2) = ins.get(ied,2) - 2*vni*ny*bs(ied,0);
			bs(ied,3) = ins.get(ied,3);
		}

		if(m->ggallfa(ied,3) == inflow_outflow_id)
		{
			//if(Mni <= -1.0)
			{
				for(int i = 0; i < NVARS; i++)
					bs(ied,i) = uinf(0,i);
			}
			/*else if(Mni > -1.0 && Mni < 0)
			{
				// subsonic inflow, specify rho and u according to FUN3D BCs paper
				for(i = 0; i < NVARS-1; i++)
					bs(ied,i) = uinf.get(0,i);
				bs(ied,3) = pi/(g-1.0) + 0.5*( uinf.get(0,1)*uinf.get(0,1) + uinf.get(0,2)*uinf.get(0,2) )/uinf.get(0,0);
			}
			else if(Mni >= 0 && Mni < 1.0)
			{
				// subsonic ourflow, specify p accoording FUN3D BCs paper
				for(i = 0; i < NVARS-1; i++)
					bs(ied,i) = ins.get(ied,i);
				bs(ied,3) = pinf/(g-1.0) + 0.5*( ins.get(ied,1)*ins.get(ied,1) + ins.get(ied,2)*ins.get(ied,2) )/ins.get(ied,0);
			}
			else
				for(i = 0; i < NVARS; i++)
					bs(ied,i) = ins.get(ied,i);*/
		}
	}
}

void EulerFV::compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs)
{
	a_real nx = m->ggallfa(ied,0);
	a_real ny = m->ggallfa(ied,1);

	a_real vni = (ins[1]*nx + ins[2]*ny)/ins[0];
	//a_real pi = (g-1.0)*(ins.get(ied,3) - 0.5*(pow(ins.get(ied,1),2)+pow(ins.get(ied,2),2))/ins.get(ied,0));
	//a_real pinf = (g-1.0)*(uinf.get(0,3) - 0.5*(pow(uinf.get(0,1),2)+pow(uinf.get(0,2),2))/uinf.get(0,0));
	//a_real ci = sqrt(g*pi/ins.get(ied,0));
	//a_real Mni = vni/ci;

	if(m->ggallfa(ied,3) == solid_wall_id)
	{
		bs[0] = ins[0];
		bs[1] = ins[1] - 2*vni*nx*bs[0];
		bs[2] = ins[2] - 2*vni*ny*bs[0];
		bs[3] = ins[3];
	}

	if(m->ggallfa(ied,3) == inflow_outflow_id)
	{
		for(int i = 0; i < NVARS; i++)
			bs[i] = uinf(0,i);
	}
}

a_real EulerFV::l2norm(const amat::Matrix<a_real>* const v)
{
	a_real norm = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		norm += v->get(iel)*v->get(iel)*m->gjacobians(iel)/2.0;
	}
	norm = sqrt(norm);
	return norm;
}

void EulerFV::compute_residual()
{
#pragma omp parallel default(shared)
	{
#pragma omp for simd
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < NVARS; i++)
				residual(iel,i) = 0.0;
			integ(iel) = 0.0;
		}

		// first, set cell-centered values of boundary cells as left-side values of boundary faces
#pragma omp for
		for(a_int ied = 0; ied < m->gnbface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			for(int ivar = 0; ivar < NVARS; ivar++)
				uleft(ied,ivar) = u.get(ielem,ivar);
		}
	}

	if(order == 2)
	{
		// get cell average values at ghost cells using BCs
		compute_boundary_states(uleft, ug);

		rec->compute_gradients();
		lim->compute_face_values();
	}
	else
	{
		// if order is 1, set the face data same as cell-centred data for all faces
		
		// set both left and right states for all interior faces
#pragma omp parallel for
		for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			a_int jelem = m->gintfac(ied,1);
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				uleft(ied,ivar) = u.get(ielem,ivar);
				uright(ied,ivar) = u.get(jelem,ivar);
			}
		}
	}

	// set right (ghost) state for boundary faces
	compute_boundary_states(uleft,uright);

	/** Compute fluxes.
	 * The integral of the maximum magnitude of eigenvalue over each face is also computed:
	 * \f[
	 * \int_{f_i} (|v_n| + c) \mathrm{d}l
	 * \f]
	 * so that time steps can be calculated for explicit time stepping.
	 */

	std::vector<a_real> ci(m->gnaface()), vni(m->gnaface()), cj(m->gnaface()), vnj(m->gnaface());

#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int ied = 0; ied < m->gnaface(); ied++)
		{
			a_real n[NDIM];
			n[0] = m->ggallfa(ied,0);
			n[1] = m->ggallfa(ied,1);
			a_real len = m->ggallfa(ied,2);

			const a_real* ulp = uleft.const_row_pointer(ied);
			const a_real* urp = uright.const_row_pointer(ied);
			a_real* fluxp = fluxes.row_pointer(ied);

			// compute flux
			inviflux->get_flux(ulp, urp, n, fluxp);

			// integrate over the face
			for(int ivar = 0; ivar < NVARS; ivar++)
					fluxp[ivar] *= len;

			//calculate presures from u
			a_real pi = (g-1)*(uleft.get(ied,3) - 0.5*(pow(uleft.get(ied,1),2)+pow(uleft.get(ied,2),2))/uleft.get(ied,0));
			a_real pj = (g-1)*(uright.get(ied,3) - 0.5*(pow(uright.get(ied,1),2)+pow(uright.get(ied,2),2))/uright.get(ied,0));
			//calculate speeds of sound
			ci[ied] = sqrt(g*pi/uleft.get(ied,0));
			cj[ied] = sqrt(g*pj/uright.get(ied,0));
			//calculate normal velocities
			vni[ied] = (uleft.get(ied,1)*n[0] +uleft.get(ied,2)*n[1])/uleft.get(ied,0);
			vnj[ied] = (uright.get(ied,1)*n[0] + uright.get(ied,2)*n[1])/uright.get(ied,0);
		}

#pragma omp barrier

		// update residual and integ
#pragma omp for
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int ied = m->gelemface(iel,ifael);
				a_real len = m->ggallfa(ied,2);
				a_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem > iel) {
					for(int ivar = 0; ivar < NVARS; ivar++)
						residual(iel,ivar) += fluxes(ied,ivar);
					integ(iel) += (fabs(vni[ied]) + ci[ied])*len;
				}
				else {
					for(int ivar = 0; ivar < NVARS; ivar++)
						residual(iel,ivar) -= fluxes(ied,ivar);
					integ(iel) += (fabs(vnj[ied]) + cj[ied])*len;
				}
			}
		}
#pragma omp barrier
#pragma omp for simd
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			dtm(iel) = m->garea(iel)/integ(iel);
		}
	} // end parallel region
}

#if HAVE_PETSC==1

void EulerFV::compute_jacobian(const bool blocked, Mat A)
{
	if(blocked)
	{
		// TODO: construct blocked Jacobian
	}
	else
	{
		Matrix<a_real>* D = new Matrix<a_real>[m->gnelem()];
		for(int iel = 0; iel < m->gnelem(); iel++) {
			D[iel].setup(NVARS,NVARS);
			D[iel].zeros();
		}

		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			a_int lelem = m->gintfac(iface,0);
			a_real n[NDIM];
			n[0] = m->ggallfa(iface,0);
			n[1] = m->ggallfa(iface,1);
			a_real len = m->ggallfa(iface,2);
			a_real uface[NVARS];
			Matrix<a_real> left(NVARS,NVARS);
			Matrix<a_real> right(NVARS,NVARS);
			
			compute_boundary_state(iface, &u(lelem,0), uface);
			jflux->get_jacobian(&u(lelem,0), uface, n, &left(0,0), &right(0,0));
			
			for(int i = 0; i < NVARS; i++)
				for(int j = 0; j < NVARS; j++) {
					left(i,j) *= len;
#pragma omp atomic write
					D[lelem](i,j) -= left(i,j);
				}
		}

		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			a_int lelem = m->gintfac(iface,0);
			a_int relem = m->gintfac(iface,1);
			a_real n[NDIM];
			n[0] = m->ggallfa(iface,0);
			n[1] = m->ggallfa(iface,1);
			a_real len = m->ggallfa(iface,2);
			a_real uface[NVARS];
			Matrix<a_real> left(NVARS,NVARS);
			Matrix<a_real> right(NVARS,NVARS);
			
			jflux->get_jacobian(&u(lelem,0), &u(relem,0), n, &left(0,0), &right(0,0));

			for(int i = 0; i < NVARS; i++)
				for(int j = 0; j < NVARS; j++) {
					left(i,j) *= len;
					right(i,j) *= len;
#pragma omp atomic write
					D[lelem](i,j) -= left(i,j);
#pragma omp atomic write
					D[relem](i,j) -= right(i,j);
				}

			PetscInt* rindices = std::malloc(NVARS*NVARS*sizeof(PetscInt));
			PetscInt* cindices = std::malloc(NVARS*NVARS*sizeof(PetscInt));
			// insert upper block U = right
			for(int i = 0; i < NVARS; i++)
				for(int j = 0; j < NVARS; j++)
				{
					rindices[i*NVARS+j] = ielem*NVARS+i;
					cindices[i*NVARS+j] = jelem*NVARS+j;
				}
			MatSetValues(A, NVARS, rindices, NVARS, cindices, &right(0,0), INSERT_VALUES);

			// insert lower block L = left
			for(int i = 0; i < NVARS; i++)
				for(int j = 0; j < NVARS; j++)
				{
					rindices[i*NVARS+j] = jelem*NVARS+i;
					cindices[i*NVARS+j] = ielem*NVARS+j;
				}
			MatSetValues(A, NVARS, rindices, NVARS, cindices, &left(0,0), INSERT_VALUES);

			std::free(rindices);
			std::free(cindices);
		}

		// diagonal blocks
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			PetscInt* rindices = std::malloc(NVARS*NVARS*sizeof(PetscInt));
			PetscInt* cindices = std::malloc(NVARS*NVARS*sizeof(PetscInt));
			
			for(int i = 0; i < NVARS; i++)
				for(int j = 0; j < NVARS; j++)
				{
					rindices[i*NVARS+j] = iel*NVARS+i;
					cindices[i*NVARS+j] = iel*NVARS+j;
				}
			MatSetValues(A, NVARS, rindices, NVARS, cindices, &D[iel](0,0), ADD_VALUES);

			std::free(rindices);
			std::free(cindices);
		}
	}
}

#else

void EulerFV::compute_jacobian(Matrix<a_real> *const D, Matrix<a_real> *const L, Matrix<a_real> *const U)
{
	for(int iel = 0; iel < m->gnelem(); iel++) {
		D[iel].zeros();
	}
	for(int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
		L[iface].zeros();
		U[iface].zeros();
	}

	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		a_real n[NDIM];
		n[0] = m->ggallfa(iface,0);
		n[1] = m->ggallfa(iface,1);
		a_real len = m->ggallfa(iface,2);
		a_real uface[NVARS];
		Matrix<a_real> left(NVARS,NVARS);
		Matrix<a_real> right(NVARS,NVARS);
		
		compute_boundary_state(iface, &u(lelem,0), uface);
		jflux->get_jacobian(&u(lelem,0), uface, n, &left(0,0), &right(0,0));
		
		for(int i = 0; i < NVARS; i++)
			for(int j = 0; j < NVARS; j++) {
				left(i,j) *= len;
#pragma omp atomic write
				D[lelem](i,j) -= left(i,j);
			}
	}

	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		a_int relem = m->gintfac(iface,1);
		a_real n[NDIM];
		n[0] = m->ggallfa(iface,0);
		n[1] = m->ggallfa(iface,1);
		a_real len = m->ggallfa(iface,2);
		
		jflux->get_jacobian(&u(lelem,0), &u(relem,0), n, &L[iface](0,0), &U[iface](0,0));

		for(int i = 0; i < NVARS; i++)
			for(int j = 0; j < NVARS; j++) {
				L[iface](i,j) *= len;
				U[iface](i,j) *= len;
#pragma omp atomic write
				D[lelem](i,j) -= L[iface](i,j);
#pragma omp atomic write
				D[relem](i,j) -= U[iface](i,j);
			}
	}
}

#endif


void EulerFV::postprocess_point()
{
	std::cout << "EulerFV: postprocess_point(): Creating output arrays...\n";
	scalars.setup(m->gnpoin(),3);
	velocities.setup(m->gnpoin(),2);
	amat::Matrix<a_real> c(m->gnpoin(),1);
	
	amat::Matrix<a_real> areasum(m->gnpoin(),1);
	amat::Matrix<a_real> up(m->gnpoin(), NVARS);
	up.zeros();
	areasum.zeros();

	int inode, ivar;
	a_int ielem, iface, ip1, ip2, ipoin;

	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(inode = 0; inode < m->gnnode(ielem); inode++)
			for(ivar = 0; ivar < NVARS; ivar++)
			{
				up(m->ginpoel(ielem,inode),ivar) += u.get(ielem,ivar)*m->garea(ielem);
				areasum(m->ginpoel(ielem,inode)) += m->garea(ielem);
			}
	}
	for(iface = 0; iface < m->gnbface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		ip1 = m->gintfac(iface,2);
		ip2 = m->gintfac(iface,3);
		for(ivar = 0; ivar < NVARS; ivar++)
		{
			up(ip1,ivar) += ug.get(iface,ivar)*m->garea(ielem);
			up(ip2,ivar) += ug.get(iface,ivar)*m->garea(ielem);
			areasum(ip1) += m->garea(ielem);
			areasum(ip2) += m->garea(ielem);
		}
	}

	for(ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(ivar = 0; ivar < NVARS; ivar++)
			up(ipoin,ivar) /= areasum(ipoin);
	
	for(ipoin = 0; ipoin < m->gnpoin(); ipoin++)
	{
		scalars(ipoin,0) = up.get(ipoin,0);
		velocities(ipoin,0) = up.get(ipoin,1)/up.get(ipoin,0);
		velocities(ipoin,1) = up.get(ipoin,2)/up.get(ipoin,0);
		//velocities(ipoin,0) = dudx(ipoin,1);
		//velocities(ipoin,1) = dudy(ipoin,1);
		a_real vmag2 = pow(velocities(ipoin,0), 2) + pow(velocities(ipoin,1), 2);
		scalars(ipoin,2) = up.get(ipoin,0)*(g-1) * (up.get(ipoin,3)/up.get(ipoin,0) - 0.5*vmag2);		// pressure
		c(ipoin) = sqrt(g*scalars(ipoin,2)/up.get(ipoin,0));
		scalars(ipoin,1) = sqrt(vmag2)/c(ipoin);
	}
	std::cout << "EulerFV: postprocess_point(): Done.\n";
}

void EulerFV::postprocess_cell()
{
	std::cout << "EulerFV: postprocess_cell(): Creating output arrays...\n";
	scalars.setup(m->gnelem(), 3);
	velocities.setup(m->gnelem(), 2);
	amat::Matrix<a_real> c(m->gnelem(), 1);

	amat::Matrix<a_real> d = u.col(0);
	scalars.replacecol(0, d);		// populate density data
	//std::cout << "EulerFV: postprocess(): Written density\n";

	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		velocities(iel,0) = u.get(iel,1)/u.get(iel,0);
		velocities(iel,1) = u.get(iel,2)/u.get(iel,0);
		//velocities(iel,0) = dudx(iel,1);
		//velocities(iel,1) = dudy(iel,1);
		a_real vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
		scalars(iel,2) = d(iel)*(g-1) * (u.get(iel,3)/d(iel) - 0.5*vmag2);		// pressure
		c(iel) = sqrt(g*scalars(iel,2)/d(iel));
		scalars(iel,1) = sqrt(vmag2)/c(iel);
	}
	std::cout << "EulerFV: postprocess_cell(): Done.\n";
}

a_real EulerFV::compute_entropy_cell()
{
	postprocess_cell();
	a_real vmaginf2 = uinf(0,1)/uinf(0,0)*uinf(0,1)/uinf(0,0) + uinf(0,2)/uinf(0,0)*uinf(0,2)/uinf(0,0);
	a_real sinf = ( uinf(0,0)*(g-1) * (uinf(0,3)/uinf(0,0) - 0.5*vmaginf2) ) / pow(uinf(0,0),g);

	amat::Matrix<a_real> s_err(m->gnelem(),1);
	a_real error = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		s_err(iel) = (scalars(iel,2)/pow(scalars(iel,0),g) - sinf)/sinf;
		error += s_err(iel)*s_err(iel)*m->gjacobians(iel)/2.0;
	}
	error = sqrt(error);

	//a_real h = sqrt((m->jacobians).max());
	a_real h = 1.0/sqrt(m->gnelem());
 
	std::cout << "EulerFV:   " << log10(h) << "  " << std::setprecision(10) << log10(error) << std::endl;

	return error;
}

amat::Matrix<a_real> EulerFV::getscalars() const
{
	return scalars;
}

amat::Matrix<a_real> EulerFV::getvelocities() const
{
	return velocities;
}

}	// end namespace
