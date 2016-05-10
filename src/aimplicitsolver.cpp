/** @file aimplicitsolver.cpp
 * @brief Implements classes for implicit solution of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date May 6, 2016
 */

#include <aimplicitsolver.hpp>

namespace acfd {

ImplicitSolver::ImplicitSolver(const UMesh2dh* const mesh, const int _order, std::string invflux, std::string reconst, std::string limiter, std::string linear_solver, 
		const double cfl_num, const double relaxation_factor)
	: cfl(cfl_num), w(relaxation_fator), order(_order), m(mesh)
{
	g = 1.4;

	std::cout << "ImplicitSolver: Setting up implicit solver for spatial order " << order << std::endl;

	// for 2D Euler equations, we have 4 variables
	nvars = 4;
	// for upto second-order finite volume, we only need 1 Guass point per face
	ngaussf = 1;

	solid_wall_id = 2;
	inflow_outflow_id = 4;

	// allocation
	m_inverse.setup(m->gnelem(),1);		// just a vector for FVM. For DG, this will be an array of Matrices
	diag = new amat::Matrix<acfd_real>[m->gnelem()];
	lambdaij.setup(m->gnaface(),1);
	residual.setup(m->gnelem(),nvars);
	u.setup(m->gnelem(), nvars);
	uinf.setup(1, nvars);
	integ.setup(m->gnelem(), 1);
	dudx.setup(m->gnelem(), nvars);
	dudy.setup(m->gnelem(), nvars);
	uleft.setup(m->gnaface(), nvars);
	uright.setup(m->gnaface(), nvars);
	rc.setup(m->gnelem(),m->gndim());
	rcg.setup(m->gnface(),m->gndim());
	ug.setup(m->gnface(),nvars);
	gr = new amat::Matrix<acfd_real>[m->gnaface()];
	for(int i = 0; i <  m->gnaface(); i++)
		gr[i].setup(ngaussf, m->gndim());

	for(int i = 0; i < m->gnelem(); i++)
	{
		m_inverse(i) = 2.0/mesh->gjacobians(i);
		diag[i].setup(nvars,nvars);
	}

	eulerflux = new FluxFunction(g);

	// set inviscid flux scheme
	if(invflux == "VANLEER")
		inviflux = new VanLeerFlux(nvars, m->gndim(), g);
	else if(invflux == "ROE")
	{
		inviflux = new RoeFlux(nvars, m->gndim(), g);
		std::cout << "ImplicitSolver: Using Roe fluxes." << std::endl;
	}
	else if(invflux == "HLLC")
	{
		inviflux = new HLLCFlux(nvars, m->gndim(), g);
		std::cout << "ImplicitSolver: Using HLLC fluxes." << std::endl;
	}
	else
		std::cout << "ImplicitSolver: ! Flux scheme not available!" << std::endl;

	// set reconstruction scheme
	std::cout << "ImplicitSolver: Reconstruction scheme is " << reconst << std::endl;
	if(reconst == "GREENGAUSS")
	{
		rec = new GreenGaussReconstruction();
		//rec->setup(m, &u, &ug, &dudx, &dudy, &rc, &rcg);
	}
	else 
	{
		rec = new WeightedLeastSquaresReconstruction();
	}
	if(order == 1) std::cout << "ImplicitSolver: No reconstruction" << std::endl;

	// set limiter
	if(limiter == "NONE")
	{
		lim = new NoLimiter(m, &u, &ug, &dudx, &dudy, &rcg, &rc, gr, &uleft, &uright);
		std::cout << "ImplicitSolver: No limiter will be used." << std::endl;
	}
	else if(limiter == "WENO")
	{
		lim = new WENOLimiter(m, &u, &ug, &dudx, &dudy, &rcg, &rc, gr, &uleft, &uright);
		std::cout << "ImplicitSolver: WENO limiter selected.\n";
	}

	// set solver
	if(linear_solver == "SSOR")
	{
		solver = new SSOR_Solver(const int num_vars, const UMesh2dh* const mesh, const Matrix<acfd_real>* const residual, const FluxFunction* const inviscid_flux,
			const Matrix<acfd_real>* const diagonal_blocks, const Matrix<acfd_real>* const lambda_ij, const Matrix<acfd_real>* const unk, const Matrix<acfd_real>* const elem_flux,
			const acfd_real omega);
		std::cout << "ImplicitSolver: SSOR solver will be used." << std::endl;
	}
}

ImplicitSolver::~ImplicitSolver()
{
	delete eulerflux;
	delete rec;
	delete inviflux;
	delete lim;
	delete [] gr;
	delete [] diag;
}

void ImplicitSolver::compute_ghost_cell_coords_about_midpoint()
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

void ImplicitSolver::compute_ghost_cell_coords_about_face()
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
void ImplicitSolver::loaddata(acfd_real Minf, acfd_real vinf, acfd_real a, acfd_real rhoinf)
{
	// Note that reference density and reference velocity are the values at infinity
	//cout << "EulerFV: loaddata(): Calculating initial data...\n";
	acfd_real vx = vinf*cos(a);
	acfd_real vy = vinf*sin(a);
	acfd_real p = rhoinf*vinf*vinf/(g*Minf*Minf);
	uinf(0,0) = rhoinf;		// should be 1
	uinf(0,1) = rhoinf*vx;
	uinf(0,2) = rhoinf*vy;
	uinf(0,3) = p/(g-1) + 0.5*rhoinf*vinf*vinf;

	//initial values are equal to boundary values
	for(int i = 0; i < m->gnelem(); i++)
		for(int j = 0; j < nvars; j++)
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
	cout << "ImplicitSolver: loaddata(): Initial data calculated.\n";
}

void ImplicitSolver::compute_boundary_states(const amat::Matrix<acfd_real>& ins, amat::Matrix<acfd_real>& bs)
{
	int lel, rel, i;
	acfd_real nx, ny, vni, pi, ci, Mni, vnj, pj, cj, Mnj, vinfx, vinfy, vinfn, vbn, pinf, pb, cinf, cb, vgx, vgy, vbx, vby;
	for(int ied = 0; ied < m->gnbface(); ied++)
	{
		nx = m->ggallfa(ied,0);
		ny = m->ggallfa(ied,1);

		vni = (ins.get(ied,1)*nx + ins.get(ied,2)*ny)/ins.get(ied,0);
		pi = (g-1.0)*(ins.get(ied,3) - 0.5*(pow(ins.get(ied,1),2)+pow(ins.get(ied,2),2))/ins.get(ied,0));
		pinf = (g-1.0)*(uinf.get(0,3) - 0.5*(pow(uinf.get(0,1),2)+pow(uinf.get(0,2),2))/uinf.get(0,0));
		ci = sqrt(g*pi/ins.get(ied,0));
		Mni = vni/ci;

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
				for(i = 0; i < nvars; i++)
					bs(ied,i) = uinf(0,i);
			}
			/*else if(Mni > -1.0 && Mni < 0)
			{
				// subsonic inflow, specify rho and u according to FUN3D BCs paper
				for(i = 0; i < nvars-1; i++)
					bs(ied,i) = uinf.get(0,i);
				bs(ied,3) = pi/(g-1.0) + 0.5*( uinf.get(0,1)*uinf.get(0,1) + uinf.get(0,2)*uinf.get(0,2) )/uinf.get(0,0);
			}
			else if(Mni >= 0 && Mni < 1.0)
			{
				// subsonic ourflow, specify p accoording FUN3D BCs paper
				for(i = 0; i < nvars-1; i++)
					bs(ied,i) = ins.get(ied,i);
				bs(ied,3) = pinf/(g-1.0) + 0.5*( ins.get(ied,1)*ins.get(ied,1) + ins.get(ied,2)*ins.get(ied,2) )/ins.get(ied,0);
			}
			else
				for(i = 0; i < nvars; i++)
					bs(ied,i) = ins.get(ied,i);*/
		}
	}
}

acfd_real ImplicitSolver::l2norm(const amat::Matrix<acfd_real>* const v)
{
	acfd_real norm = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		norm += v->get(iel)*v->get(iel)*m->gjacobians(iel)/2.0;
	}
	norm = sqrt(norm);
	return norm;
}

void ImplicitSolver::compute_RHS()
{
	residual.zeros();
	
	int ied, ielem, jelem, ivar;

	// first, set cell-centered values of boundary cells as left-side values of boundary faces
	for(ied = 0; ied < m->gnbface(); ied++)
	{
		ielem = m->gintfac(ied,0);
		for(ivar = 0; ivar < nvars; ivar++)
			uleft(ied,ivar) = u.get(ielem,ivar);
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
		for(ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			ielem = m->gintfac(ied,0);
			jelem = m->gintfac(ied,1);
			for(ivar = 0; ivar < nvars; ivar++)
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
	acfd_real *n, nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj, vmags;
	int lel, rel;
	n = new acfd_real[m->gndim()];
	amat::Matrix<acfd_real> ul(nvars,1), ur(nvars,1), flux(nvars,1);

	for(ied = 0; ied < m->gnaface(); ied++)
	{
		lel = m->gintfac(ied,0);	// left element
		rel = m->gintfac(ied,1);	// right element

		n[0] = m->ggallfa(ied,0);
		n[1] = m->ggallfa(ied,1);
		len = m->ggallfa(ied,2);

		for(ivar = 0; ivar < nvars; ivar++)
		{
			ul(ivar) = uleft.get(ied,ivar);
			ur(ivar) = uright.get(ied,ivar);
		}

		// compute flux
		inviflux->get_flux(&ul, &ur, n, &flux);

		// integrate over the face
		for(ivar = 0; ivar < nvars; ivar++)
				flux(ivar) *= len;

		// scatter the flux to elements' residuals
		for(ivar = 0; ivar < nvars; ivar++)
		{
			residual(lel,ivar) -= flux(ivar);
			if(rel >= 0 && rel < m->gnelem())
				residual(rel,ivar) += flux(ivar);
		}

		//calculate presures from u
		pi = (g-1)*(uleft.get(ied,3) - 0.5*(pow(uleft.get(ied,1),2)+pow(uleft.get(ied,2),2))/uleft.get(ied,0));
		pj = (g-1)*(uright.get(ied,3) - 0.5*(pow(uright.get(ied,1),2)+pow(uright.get(ied,2),2))/uright.get(ied,0));
		//calculate speeds of sound
		ci = sqrt(g*pi/uleft.get(ied,0));
		cj = sqrt(g*pj/uright.get(ied,0));
		//calculate normal velocities
		vni = (uleft.get(ied,1)*n[0] +uleft.get(ied,2)*n[1])/uleft.get(ied,0);
		vnj = (uright.get(ied,1)*n[0] + uright.get(ied,2)*n[1])/uright.get(ied,0);

		// calculate integ for CFL purposes
		integ(lel,0) += (fabs(vni) + ci)*len;
		if(rel >= 0 && rel < m->gnelem())
			integ(rel,0) += (fabs(vnj) + cj)*len;
	}
	delete [] n;
}

void ImplicitSolver::compute_LHS()
{
	// compute diagonal blocks
}

void ImplicitSolver::postprocess_point()
{
	cout << "ImplicitSolver: postprocess_point(): Creating output arrays...\n";
	scalars.setup(m->gnpoin(),3);
	velocities.setup(m->gnpoin(),2);
	amat::Matrix<acfd_real> c(m->gnpoin(),1);
	
	amat::Matrix<acfd_real> areasum(m->gnpoin(),1);
	amat::Matrix<acfd_real> up(m->gnpoin(), nvars);
	up.zeros();
	areasum.zeros();

	int inode, ivar;
	acfd_int ielem, iface, ip1, ip2, ipoin;

	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(inode = 0; inode < m->gnnode(ielem); inode++)
			for(ivar = 0; ivar < nvars; ivar++)
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
		for(ivar = 0; ivar < nvars; ivar++)
		{
			up(ip1,ivar) += ug.get(iface,ivar)*m->garea(ielem);
			up(ip2,ivar) += ug.get(iface,ivar)*m->garea(ielem);
			areasum(ip1) += m->garea(ielem);
			areasum(ip2) += m->garea(ielem);
		}
	}

	for(ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(ivar = 0; ivar < nvars; ivar++)
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
	cout << "EulerFV: postprocess_point(): Done.\n";
}

void ImplicitSolver::postprocess_cell()
{
	cout << "ImplicitSolver: postprocess_cell(): Creating output arrays...\n";
	scalars.setup(m->gnelem(), 3);
	velocities.setup(m->gnelem(), 2);
	amat::Matrix<acfd_real> c(m->gnelem(), 1);

	amat::Matrix<acfd_real> d = u.col(0);
	scalars.replacecol(0, d);		// populate density data
	//cout << "EulerFV: postprocess(): Written density\n";

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
	cout << "EulerFV: postprocess_cell(): Done.\n";
}

acfd_real ImplicitSolver::compute_entropy_cell()
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
 
	cout << "EulerFV:   " << log(h) << "  " << setprecision(10) << log(error) << endl;

	return error;
}

amat::Matrix<acfd_real> ImplicitSolver::getscalars() const
{
	return scalars;
}

amat::Matrix<acfd_real> ImplicitSolver::getvelocities() const
{
	return velocities;
}

SteadyStateImplicitSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter, std::string linear_solver, const double cfl, const double omega,
		const acfd_real lin_tol, const int lin_maxiter, const acfd_real steady_tol, const int steady_maxiter)
	: ImplicitSolver(mesh, invflux, reconst, limiter, linear_solver, cfl, omega), lintol(lin_tol), linmaxiter(lin_maxiter), steadytol(steady_tol), steadymaxiter(steady_maxiter)
{ }

void SteadyStateImplicitSolver::solve()
{
	int step = 0;
	acfd_real resi = 1.0;
	acfd_real initres = 1.0;
	amat::Matrix<acfd_real>* err;
	err = new amat::Matrix<acfd_real>[nvars];
	for(int i = 0; i<nvars; i++)
		err[i].setup(m->gnelem(),1);
	amat::Matrix<acfd_real> res(nvars,1);
	res.ones();
	amat::Matrix<acfd_real> dtm(m->gnelem(), 1);		// for local time-stepping
	amat::Matrix<acfd_real> uold(u.rows(), u.cols());

	while(resi/initres > steadytol && step < steadymaxiter)
	{
		//cout << "EulerFV: solve_rk1_steady(): Entered loop. Step " << step << endl;
		// reset fluxes
		integ.zeros();		// reset CFL data

		//calculate fluxes
		compute_RHS();		// this invokes Flux calculating function after zeroing the residuals

		/*if(step==0) {
			//dudx.mprint(); dudy.mprint();
			break;
		}*/

		//calculate dt based on CFL

		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
		}

		uold = u;
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
			{
				u(iel,i) += dtm.get(iel)*m_inverse.get(iel)*residual.get(iel,i);
			}
		}

		//if(step == 0) { dudx.mprint();  break; }

		for(int i = 0; i < nvars; i++)
		{
			err[i] = (u-uold).col(i);
			//err[i] = residual.col(i);
			res(i) = l2norm(&err[i]);
		}
		resi = res.max();

		if(step == 0)
			initres = resi;

		if(step % 20 == 0)
			std::cout << "EulerFV: solve_rk1_steady(): Step " << step << ", rel residual " << resi/initres << std::endl;

		step++;
		/*acfd_real totalenergy = 0;
		for(int i = 0; i < m->gnelem(); i++)
			totalenergy += u(i,3)*m->jacobians(i);
		cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
		//if(step == 10000) break;
	}

	if(step == maxiter)
		std::cout << "ImplicitSolver: solve_rk1_steady(): Exceeded max iterations!" << std::endl;
	//calculate gradients
	//rec->compute_gradients();
	delete [] err;
}
}	// end namespace
