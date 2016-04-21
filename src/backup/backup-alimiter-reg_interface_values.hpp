    void compute_reg_interface_values()
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

                    (*ufl)(ied,i) = u->get(ielem,i) + phi_l->get(ied,i)*dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + phi_l->get(ied,i)*dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));
    				(*ufr)(ied,i) = u->get(jelem,i) + phi_r->get(ied,i)*dudx->get(jelem,i)*(gx->get(ied,0)-xi->get(jelem)) + phi_r->get(ied,i)*dudy->get(jelem,i)*(gy->get(ied,0)-yi->get(jelem));
    			}
            }
		}
		//cout << "VanAlbadaLimiter: compute_interface_values(): Computing values at faces - boundary\n";
		//Now calculate ghost states at boundary faces using the ufl and ufr of cells
		// (b) boundary faces
		for(int ied = 0; ied < m->gnbface(); ied++)
		{
			int ielem = m->gintfac(ied,0); int lel = ielem;
			double nx = m->ggallfa(ied,0);
			double ny = m->ggallfa(ied,1);

            for(int ig = 0; ig < ng; ig++)
            {
                Matrix<double> deltam(nvars,1,ROWMAJOR);

                for(int i = 0; i < nvars; i++)
                {

                    (*ufl)(ied,i) = u->get(ielem,i) + phi_l->get(ied,i)*dudx->get(ielem,i)*(gx->get(ied,0)-xi->get(ielem)) + phi_l->get(ied,i)*dudy->get(ielem,i)*(gy->get(ied,0)-yi->get(ielem));
    			}

    			double vni = (ufl->get(ied,1)*nx + ufl->get(ied,2)*ny)/ufl->get(ied,0);
    			double pi = (g-1)*(ufl->get(ied,3) - 0.5*(pow(ufl->get(ied,1),2)+pow(ufl->get(ied,2),2))/ufl->get(ied,0));
    			double ci = sqrt(g*pi/ufl->get(ied,0));
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
    					pj = (g-1)*(ufr(ied,3) - 0.5*(pow(ufr(ied,1),2)+pow(ufr(ied,2),2))/ufr(ied,0));
    					cj = sqrt(g*pj/ufr(ied,0));
    					vnj = (ufr(ied,1)*nx + ufr(ied,2)*ny)/ufr(ied,0);
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
    					pj = (g-1)*(ufr(ied,3) - 0.5*(pow(ufr(ied,1),2)+pow(ufr(ied,2),2))/ufr(ied,0));
    					cj = sqrt(g*pj/ufr(ied,0));
    					vnj = (ufr(ied,1)*nx + ufr(ied,2)*ny)/ufr(ied,0);
    				} */

    				// Naive way
    				for(int i = 0; i < nvars; i++)
    					(*ufr)(ied,i) = uinf->get(0,i);
    			}
            }
		}
    }
