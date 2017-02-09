/** @file aexplicitsolver.cpp
 * @brief Implements a driver class for explicit solution of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 */

#include <aexplicitsolver.hpp>

namespace acfd {

ExplicitSolver::ExplicitSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter)
{
	m = mesh;
	order = _order;
	g = 1.4;

	std::cout << "ExplicitSolver: Setting up explicit solver for spatial order " << order << std::endl;

	// for 2D Euler equations, we have 4 variables
	nvars = NVARS;
	// for upto second-order finite volume, we only need 1 Guass point per face
	ngaussf = 1;

	/// TODO: Take the two values below as input from control file, rather than hardcoding
	solid_wall_id = 2;
	inflow_outflow_id = 4;

	// allocation
	m_inverse.setup(m->gnelem(),1);		// just a vector for FVM. For DG, this will be an array of Matrices
	residual.setup(m->gnelem(),NVARS);
	u.setup(m->gnelem(), NVARS);
	uinf.setup(1, NVARS);
	integ.setup(m->gnelem(), 1);
	dudx.setup(m->gnelem(), NVARS);
	dudy.setup(m->gnelem(), NVARS);
	fluxes.setup(m->gnaface(), NVARS);
	uleft.setup(m->gnaface(), NVARS);
	uright.setup(m->gnaface(), NVARS);
	rc.setup(m->gnelem(),m->gndim());
	rcg.setup(m->gnface(),m->gndim());
	ug.setup(m->gnface(),NVARS);
	gr = new amat::Matrix<acfd_real>[m->gnaface()];
	for(int i = 0; i <  m->gnaface(); i++)
		gr[i].setup(ngaussf, m->gndim());

	for(int i = 0; i < m->gnelem(); i++)
		m_inverse(i) = 2.0/mesh->gjacobians(i);

	// set inviscid flux scheme
	if(invflux == "VANLEER")
		inviflux = new VanLeerFlux(NVARS, m->gndim(), g);
	else if(invflux == "ROE")
	{
		inviflux = new RoeFlux(NVARS, m->gndim(), g);
		std::cout << "ExplicitSolver: Using Roe fluxes." << std::endl;
	}
	else if(invflux == "HLLC")
	{
		inviflux = new HLLCFlux(NVARS, m->gndim(), g);
		std::cout << "ExplicitSolver: Using HLLC fluxes." << std::endl;
	}
	else
		std::cout << "ExplicitSolver: ! Flux scheme not available!" << std::endl;

	// set reconstruction scheme
	std::cout << "ExplicitSolver: Reconstruction scheme is " << reconst << std::endl;
	if(reconst == "GREENGAUSS")
	{
		rec = new GreenGaussReconstruction();
		//rec->setup(m, &u, &ug, &dudx, &dudy, &rc, &rcg);
	}
	else 
	{
		rec = new WeightedLeastSquaresReconstruction();
	}
	if(order == 1) std::cout << "ExplicitSolver: No reconstruction" << std::endl;

	// set limiter
	if(limiter == "NONE")
	{
		lim = new NoLimiter(m, &u, &ug, &dudx, &dudy, &rcg, &rc, gr, &uleft, &uright);
		std::cout << "ExplicitSolver: No limiter will be used." << std::endl;
	}
	else if(limiter == "WENO")
	{
		lim = new WENOLimiter(m, &u, &ug, &dudx, &dudy, &rcg, &rc, gr, &uleft, &uright);
		std::cout << "ExplicitSolver: WENO limiter selected.\n";
	}
}

ExplicitSolver::~ExplicitSolver()
{
	delete rec;
	delete inviflux;
	delete lim;
	delete [] gr;
}

void ExplicitSolver::compute_ghost_cell_coords_about_midpoint()
{
	int iface, ielem, idim, ip1, ip2;
	std::vector<acfd_real> midpoint(m->gndim());
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

void ExplicitSolver::compute_ghost_cell_coords_about_face()
{
	int ied, ig, ielem;
	acfd_real x1, y1, x2, y2, xs, ys, xi, yi;

	for(ied = 0; ied < m->gnbface(); ied++)
	{
		ielem = m->gintfac(ied,0); //int lel = ielem;
		//jelem = m->gintfac(ied,1); //int rel = jelem;
		acfd_real nx = m->ggallfa(ied,0);
		acfd_real ny = m->ggallfa(ied,1);

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
void ExplicitSolver::loaddata(acfd_real Minf, acfd_real vinf, acfd_real a, acfd_real rhoinf)
{
	// Note that reference density and reference velocity are the values at infinity
	//std::cout << "EulerFV: loaddata(): Calculating initial data...\n";
	acfd_real vx = vinf*cos(a);
	acfd_real vy = vinf*sin(a);
	acfd_real p = rhoinf*vinf*vinf/(g*Minf*Minf);
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
			rc(ielem,idim) = rc(ielem,idim) / (acfd_real)(m->gnnode(ielem));
		}
	}

	int ied, ig, ielem;
	acfd_real x1, y1, x2, y2, xs, ys, xi, yi;

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
			gr[ied](ig,0) = x1 + (acfd_real)(ig+1.0)/(acfd_real)(ngaussf+1.0) * (x2-x1);
			gr[ied](ig,1) = y1 + (acfd_real)(ig+1.0)/(acfd_real)(ngaussf+1.0) * (y2-y1);
		}
	}

	rec->setup(m, &u, &ug, &dudx, &dudy, &rc, &rcg);
	std::cout << "ExplicitSolver: loaddata(): Initial data calculated.\n";
}

void ExplicitSolver::compute_boundary_states(const amat::Matrix<acfd_real>& ins, amat::Matrix<acfd_real>& bs)
{
#pragma omp parallel for default(shared)
	for(int ied = 0; ied < m->gnbface(); ied++)
	{
		acfd_real nx = m->ggallfa(ied,0);
		acfd_real ny = m->ggallfa(ied,1);

		acfd_real vni = (ins.get(ied,1)*nx + ins.get(ied,2)*ny)/ins.get(ied,0);
		acfd_real pi = (g-1.0)*(ins.get(ied,3) - 0.5*(pow(ins.get(ied,1),2)+pow(ins.get(ied,2),2))/ins.get(ied,0));
		acfd_real pinf = (g-1.0)*(uinf.get(0,3) - 0.5*(pow(uinf.get(0,1),2)+pow(uinf.get(0,2),2))/uinf.get(0,0));
		acfd_real ci = sqrt(g*pi/ins.get(ied,0));
		acfd_real Mni = vni/ci;

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

acfd_real ExplicitSolver::l2norm(const amat::Matrix<acfd_real>* const v)
{
	acfd_real norm = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		norm += v->get(iel)*v->get(iel)*m->gjacobians(iel)/2.0;
	}
	norm = sqrt(norm);
	return norm;
}

void ExplicitSolver::compute_RHS()
{
	//std::cout << "Computing res ---\n";
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
		for(acfd_int ied = 0; ied < m->gnbface(); ied++)
		{
			acfd_int ielem = m->gintfac(ied,0);
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
		for(acfd_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			acfd_int ielem = m->gintfac(ied,0);
			acfd_int jelem = m->gintfac(ied,1);
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

	std::vector<acfd_real> ci(m->gnaface()), vni(m->gnaface()), cj(m->gnaface()), vnj(m->gnaface());

#pragma omp parallel default(shared)
	{
#pragma omp for
		for(acfd_int ied = 0; ied < m->gnaface(); ied++)
		{
			//acfd_int lel = m->gintfac(ied,0);	// left element
			//acfd_int rel = m->gintfac(ied,1);	// right element

			acfd_real n[NDIM];
			n[0] = m->ggallfa(ied,0);
			n[1] = m->ggallfa(ied,1);
			acfd_real len = m->ggallfa(ied,2);

			const acfd_real* ulp = uleft.const_row_pointer(ied);
			const acfd_real* urp = uright.const_row_pointer(ied);
			acfd_real* fluxp = fluxes.row_pointer(ied);

			// compute flux
			inviflux->get_flux(ulp, urp, n, fluxp);

			// integrate over the face
			for(int ivar = 0; ivar < NVARS; ivar++)
					fluxp[ivar] *= len;

			// scatter the flux to elements' residuals
			/*for(int ivar = 0; ivar < NVARS; ivar++)
			{
				residual(lel,ivar) -= fluxp[ivar];
				if(rel >= 0 && rel < m->gnelem())
					residual(rel,ivar) += fluxp[ivar];
			}*/

			//calculate presures from u
			acfd_real pi = (g-1)*(uleft.get(ied,3) - 0.5*(pow(uleft.get(ied,1),2)+pow(uleft.get(ied,2),2))/uleft.get(ied,0));
			acfd_real pj = (g-1)*(uright.get(ied,3) - 0.5*(pow(uright.get(ied,1),2)+pow(uright.get(ied,2),2))/uright.get(ied,0));
			//calculate speeds of sound
			ci[ied] = sqrt(g*pi/uleft.get(ied,0));
			cj[ied] = sqrt(g*pj/uright.get(ied,0));
			//calculate normal velocities
			vni[ied] = (uleft.get(ied,1)*n[0] +uleft.get(ied,2)*n[1])/uleft.get(ied,0);
			vnj[ied] = (uright.get(ied,1)*n[0] + uright.get(ied,2)*n[1])/uright.get(ied,0);

			// calculate integ for CFL purposes
			/*integ(lel,0) += (fabs(vni) + ci)*len;
			if(rel >= 0 && rel < m->gnelem())
				integ(rel,0) += (fabs(vnj) + cj)*len;*/
		}

		// update residual and integ
		//std::cout << "Beginning new loop --- \n";
#pragma omp for
		for(acfd_int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				acfd_int ied = m->gelemface(iel,ifael);
				acfd_real len = m->ggallfa(ied,2);
				acfd_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem > iel) {
					for(int ivar = 0; ivar < NVARS; ivar++)
						residual(iel,ivar) -= fluxes(ied,ivar);
					integ(iel) += (fabs(vni[ied]) + ci[ied])*len;
				}
				else {
					for(int ivar = 0; ivar < NVARS; ivar++)
						residual(iel,ivar) += fluxes(ied,ivar);
					integ(iel) += (fabs(vnj[ied]) + cj[ied])*len;
				}
			}
		}
	} // end parallel region
}

void ExplicitSolver::solve_rk1_steady(const acfd_real tol, const int maxiter, const acfd_real cfl)
{
	int step = 0;
	acfd_real resi = 1.0;
	acfd_real initres = 1.0;
	amat::Matrix<acfd_real> res(NVARS,1);
	res.ones();
	amat::Matrix<acfd_real> dtm(m->gnelem(), 1);		// for local time-stepping
	amat::Matrix<acfd_real> uold(u.rows(), u.cols());

	while(resi/initres > tol && step < maxiter)
	{
		//std::cout << "EulerFV: solve_rk1_steady(): Entered loop. Step " << step << std::endl;

		//calculate fluxes
		compute_RHS();		// this invokes Flux calculating function after zeroing the residuals, also computes max wave speeds integ

		acfd_real err[NVARS];
		acfd_real errmass = 0;
		for(int i = 0; i < NVARS; i++)
			err[i] = 0;

		//calculate dt based on CFL
#pragma omp parallel default(shared)
		{
#pragma omp for simd
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

#pragma omp for simd
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < NVARS; i++)
				{
					//uold(iel,i) = u(iel,i);
					u(iel,i) += dtm.get(iel)*m_inverse.get(iel)*residual.get(iel,i);
				}
			}

#pragma omp for simd reduction(+:errmass)
//#pragma omp for simd reduction(+:err[:NVARS])
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				/*for(int i = 0; i < NVARS; i++)
				{
					err[i] += residual(iel,i)*residual(iel,i)*m->garea(iel);
				}*/
				errmass += residual(iel,0)*residual(iel,0)*m->garea(iel);
			}
		} // end parallel region

		/*resi = 2e-15;
		for(int i = 0; i < NVARS; i++)
			if(err[i] > resi*resi)
				resi = err[i];
		resi = sqrt(resi);*/
		resi = sqrt(errmass);

		if(step == 0)
			initres = resi;

		if(step % 50 == 0)
			std::cout << "EulerFV: solve_rk1_steady(): Step " << step << ", rel residual " << resi/initres << std::endl;

		step++;
		/*acfd_real totalenergy = 0;
		for(int i = 0; i < m->gnelem(); i++)
			totalenergy += u(i,3)*m->jacobians(i);
			std::cout << "EulerFV: solve(): Total energy = " << totalenergy << std::endl;*/
		//if(step == 10000) break;
	}

	if(step == maxiter)
		std::cout << "ExplicitSolver: solve_rk1_steady(): Exceeded max iterations!" << std::endl;
}

void ExplicitSolver::postprocess_point()
{
	std::cout << "ExplicitSolver: postprocess_point(): Creating output arrays...\n";
	scalars.setup(m->gnpoin(),3);
	velocities.setup(m->gnpoin(),2);
	amat::Matrix<acfd_real> c(m->gnpoin(),1);
	
	amat::Matrix<acfd_real> areasum(m->gnpoin(),1);
	amat::Matrix<acfd_real> up(m->gnpoin(), NVARS);
	up.zeros();
	areasum.zeros();

	int inode, ivar;
	acfd_int ielem, iface, ip1, ip2, ipoin;

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
		acfd_real vmag2 = pow(velocities(ipoin,0), 2) + pow(velocities(ipoin,1), 2);
		scalars(ipoin,2) = up.get(ipoin,0)*(g-1) * (up.get(ipoin,3)/up.get(ipoin,0) - 0.5*vmag2);		// pressure
		c(ipoin) = sqrt(g*scalars(ipoin,2)/up.get(ipoin,0));
		scalars(ipoin,1) = sqrt(vmag2)/c(ipoin);
	}
	std::cout << "EulerFV: postprocess_point(): Done.\n";
}

void ExplicitSolver::postprocess_cell()
{
	std::cout << "ExplicitSolver: postprocess_cell(): Creating output arrays...\n";
	scalars.setup(m->gnelem(), 3);
	velocities.setup(m->gnelem(), 2);
	amat::Matrix<acfd_real> c(m->gnelem(), 1);

	amat::Matrix<acfd_real> d = u.col(0);
	scalars.replacecol(0, d);		// populate density data
	//std::cout << "EulerFV: postprocess(): Written density\n";

	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		velocities(iel,0) = u.get(iel,1)/u.get(iel,0);
		velocities(iel,1) = u.get(iel,2)/u.get(iel,0);
		//velocities(iel,0) = dudx(iel,1);
		//velocities(iel,1) = dudy(iel,1);
		acfd_real vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
		scalars(iel,2) = d(iel)*(g-1) * (u.get(iel,3)/d(iel) - 0.5*vmag2);		// pressure
		c(iel) = sqrt(g*scalars(iel,2)/d(iel));
		scalars(iel,1) = sqrt(vmag2)/c(iel);
	}
	std::cout << "EulerFV: postprocess_cell(): Done.\n";
}

acfd_real ExplicitSolver::compute_entropy_cell()
{
	postprocess_cell();
	acfd_real vmaginf2 = uinf(0,1)/uinf(0,0)*uinf(0,1)/uinf(0,0) + uinf(0,2)/uinf(0,0)*uinf(0,2)/uinf(0,0);
	acfd_real sinf = ( uinf(0,0)*(g-1) * (uinf(0,3)/uinf(0,0) - 0.5*vmaginf2) ) / pow(uinf(0,0),g);

	amat::Matrix<acfd_real> s_err(m->gnelem(),1);
	acfd_real error = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		s_err(iel) = (scalars(iel,2)/pow(scalars(iel,0),g) - sinf)/sinf;
		error += s_err(iel)*s_err(iel)*m->gjacobians(iel)/2.0;
	}
	error = sqrt(error);

	//acfd_real h = sqrt((m->jacobians).max());
	acfd_real h = 1.0/sqrt(m->gnelem());
 
	std::cout << "EulerFV:   " << log10(h) << "  " << std::setprecision(10) << log10(error) << std::endl;

	return error;
}

amat::Matrix<acfd_real> ExplicitSolver::getscalars() const
{
	return scalars;
}

amat::Matrix<acfd_real> ExplicitSolver::getvelocities() const
{
	return velocities;
}

}	// end namespace
