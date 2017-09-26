

/*
  Differentiation of get_HLLC_flux in forward (tangent) mode:
   variations   of useful results: *flux
   with respect to varying inputs: *ur
   RW status of diff variables: *flux:out *ur:in
   Plus diff mem management of: flux:in ur:in
   Multidirectional mode

 * Currently, the estimated signal speeds are the Einfeldt estimates, 
 * not the corrected ones given by Remaki et. al.
 */
void HLLCFlux::getFluxJac_right(const a_real *const ul, const a_real *const ur, 
		const a_real *const n, 
		a_real *const __restrict flux, a_real *const __restrict fluxd) 
{
    a_real urd[NVARS][NVARS];
	for(int i = 0; i < NVARS; i++) {
		for(int j = 0; j < NVARS; j++)
			urd[i][j] = 0;
		urd[i][i] = 1.0;
	}
	
    a_real utemp[NVARS];
    a_real utempd[NVARS][NVARS];
    int ii1;
    const a_real vxi = ul[1]/ul[0];
    const a_real vyi = ul[2]/ul[0];
    a_real vxjd[NVARS];
    const a_real vxj = ur[1]/ur[0];
    a_real vyjd[NVARS];
    const a_real vyj = ur[2]/ur[0];
    const a_real vmag2i = vxi*vxi + vyi*vyi;
    a_real vmag2jd[NVARS];
    const a_real vmag2j = vxj*vxj + vyj*vyj;
    const a_real pi = (g-1.0)*(ul[3]-0.5*ul[0]*vmag2i);
    a_real pjd[NVARS];
    const a_real pj = (g-1.0)*(ur[3]-0.5*ur[0]*vmag2j);
    a_real arg1d[NVARS];
    int nd;
    a_real arg1;
    arg1 = g*pi/ul[0];
    const a_real ci = sqrt(arg1);
    arg1 = g*pj/ur[0];
    a_real vnjd[NVARS];
    a_real cjd[NVARS];
    a_real Hjd[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        vxjd[nd] = (urd[1][nd]*ur[0]-ur[1]*urd[0][nd])/(ur[0]*ur[0]);
        vyjd[nd] = (urd[2][nd]*ur[0]-ur[2]*urd[0][nd])/(ur[0]*ur[0]);
        vnjd[nd] = n[0]*vxjd[nd] + n[1]*vyjd[nd];
        vmag2jd[nd] = vxjd[nd]*vxj + vxj*vxjd[nd] + vyjd[nd]*vyj + vyj*vyjd[nd];
        pjd[nd] = (g-1.0)*(urd[3][nd]-0.5*(urd[0][nd]*vmag2j+ur[0]*vmag2jd[nd]));
        arg1d[nd] = (g*pjd[nd]*ur[0]-g*pj*urd[0][nd])/(ur[0]*ur[0]);
        cjd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
        Hjd[nd] = ((urd[3][nd]+pjd[nd])*ur[0]-(ur[3]+pj)*urd[0][nd])/(ur[0]*ur[0]);
        arg1d[nd] = urd[0][nd]/ul[0];
    }
    const a_real vni = vxi*n[0] + vyi*n[1];
    const a_real vnj = vxj*n[0] + vyj*n[1];
    const a_real cj = sqrt(arg1);
    const a_real Hi = (ul[3]+pi)/ul[0];
    const a_real Hj = (ur[3]+pj)/ur[0];
    arg1 = ur[0]/ul[0];
    a_real Rijd[NVARS];
    const a_real Rij = sqrt(arg1);
    a_real vxijd[NVARS];
    const a_real vxij = (Rij*vxj+vxi)/(Rij+1.0);
    a_real vyijd[NVARS];
    const a_real vyij = (Rij*vyj+vyi)/(Rij+1.0);
    a_real Hijd[NVARS];
    a_real vm2ijd[NVARS];
    a_real vnijd[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        Rijd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
        vxijd[nd] = ((Rijd[nd]*vxj+Rij*vxjd[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((Rijd[nd]*vyj+Rij*vyjd[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        Hijd[nd] = ((Rijd[nd]*Hj+Rij*Hjd[nd])*(Rij+1.0)-(Rij*Hj+Hi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vm2ijd[nd] = vxijd[nd]*vxij + vxij*vxijd[nd] + vyijd[nd]*vyij + vyij*vyijd[nd];
        vnijd[nd] = n[0]*vxijd[nd] + n[1]*vyijd[nd];
        arg1d[nd] = (g-1.0)*(Hijd[nd]-0.5*vm2ijd[nd]);
    }
    const a_real Hij = (Rij*Hj+Hi)/(Rij+1.0);
    const a_real vm2ij = vxij*vxij + vyij*vyij;
    const a_real vnij = vxij*n[0] + vyij*n[1];
    arg1 = (g-1.0)*(Hij-vm2ij*0.5);
    a_real cijd[NVARS];
    for (nd = 0; nd < NVARS; ++nd)
        cijd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
    const a_real cij = sqrt(arg1);
    // estimate signal speeds (classical; not Remaki corrected)
    a_real sr, sl;
    a_real srd[NVARS], sld[NVARS];
    sl = vni - ci;
    if (sl > vnij - cij) {
        for (nd = 0; nd < NVARS; ++nd)
            sld[nd] = vnijd[nd] - cijd[nd];
        sl = vnij - cij;
    } else
        for (nd = 0; nd < NVARS; ++nd)
            sld[nd] = 0.0;
    for (nd = 0; nd < NVARS; ++nd)
        srd[nd] = vnjd[nd] + cjd[nd];
    sr = vnj + cj;
    if (sr < vnij + cij) {
        for (nd = 0; nd < NVARS; ++nd)
            srd[nd] = vnijd[nd] + cijd[nd];
        sr = vnij + cij;
    }
    a_real sm;
    a_real smd[NVARS];
    for (nd = 0; nd < NVARS; ++nd)
        smd[nd] = (((urd[0][nd]*vnj+ur[0]*vnjd[nd])*(sr-vnj)+ur[0]*vnj*(srd[nd
            ]-vnjd[nd])-ul[0]*vni*sld[nd]-pjd[nd])*(ur[0]*(sr-vnj)-ul[0]*(sl-
            vni))-(ur[0]*vnj*(sr-vnj)-ul[0]*vni*(sl-vni)+pi-pj)*(urd[0][nd]*(
            sr-vnj)+ur[0]*(srd[nd]-vnjd[nd])-ul[0]*sld[nd]))/((ur[0]*(sr-vnj)-
            ul[0]*(sl-vni))*(ur[0]*(sr-vnj)-ul[0]*(sl-vni)));
    sm = (ur[0]*vnj*(sr-vnj)-ul[0]*vni*(sl-vni)+pi-pj)/(ur[0]*(sr-vnj)-ul[0]*(
        sl-vni));
    // compute fluxes
    if (sl > 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = 0.0;
        flux[0] = vni*ul[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = 0.0;
        flux[1] = vni*ul[1] + pi*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = 0.0;
        flux[2] = vni*ul[2] + pi*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = 0.0;
        flux[3] = vni*(ul[3]+pi);
    } else if (sl <= 0 && sm > 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = 0.0;
        flux[0] = vni*ul[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = 0.0;
        flux[1] = vni*ul[1] + pi*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = 0.0;
        flux[2] = vni*ul[2] + pi*n[1];
        for (nd = 0; nd < NVARS; ++nd) {
            fluxd[3*NVARS+nd] = 0.0;
            for (ii1 = 0; ii1 < NVARS; ++ii1)
                utempd[ii1][nd] = 0.0;
        }
        flux[3] = vni*(ul[3]+pi);
        a_real pstar;
        a_real pstard[NVARS];
        for (nd = 0; nd < NVARS; ++nd) {
            pstard[nd] = ul[0]*(-(sld[nd]*(vni-sm))-(vni-sl)*smd[nd]);
            utempd[0][nd] = (ul[0]*sld[nd]*(sl-sm)-ul[0]*(sl-vni)*(sld[nd]-smd
                [nd]))/((sl-sm)*(sl-sm));
        }
        pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
        utemp[0] = ul[0]*(sl-vni)/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[1][nd] = ((ul[1]*sld[nd]+n[0]*pstard[nd])*(sl-sm)-((sl-vni)
                *ul[1]+(pstar-pi)*n[0])*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        utemp[1] = ((sl-vni)*ul[1]+(pstar-pi)*n[0])/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[2][nd] = ((ul[2]*sld[nd]+n[1]*pstard[nd])*(sl-sm)-((sl-vni)
                *ul[2]+(pstar-pi)*n[1])*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        utemp[2] = ((sl-vni)*ul[2]+(pstar-pi)*n[1])/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[3][nd] = ((ul[3]*sld[nd]+pstard[nd]*sm+pstar*smd[nd])*(sl-
                sm)-((sl-vni)*ul[3]-pi*vni+pstar*sm)*(sld[nd]-smd[nd]))/((sl-
                sm)*(sl-sm));
        utemp[3] = ((sl-vni)*ul[3]-pi*vni+pstar*sm)/(sl-sm);
        for (int ivar = 0; ivar < NVARS; ++ivar) {
            for (nd = 0; nd < NVARS; ++nd)
                fluxd[ivar*NVARS+nd] = fluxd[ivar*NVARS+nd] + sld[nd]*(utemp[ivar]-ul[
                    ivar]) + sl*utempd[ivar][nd];
            flux[ivar] += sl*(utemp[ivar]-ul[ivar]);
        }
    } else if (sm <= 0 && sr >= 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = vnjd[nd]*ur[0] + vnj*urd[0][nd];
        flux[0] = vnj*ur[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = vnjd[nd]*ur[1] + vnj*urd[1][nd] + n[0]*pjd[nd];
        flux[1] = vnj*ur[1] + pj*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = vnjd[nd]*ur[2] + vnj*urd[2][nd] + n[1]*pjd[nd];
        flux[2] = vnj*ur[2] + pj*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = vnjd[nd]*(ur[3]+pj) + vnj*(urd[3][nd]+pjd[nd]);
        flux[3] = vnj*(ur[3]+pj);
        a_real pstar;
        a_real pstard[NVARS];
        for (nd = 0; nd < NVARS; ++nd) {
            pstard[nd] = (urd[0][nd]*(vnj-sr)+ur[0]*(vnjd[nd]-srd[nd]))*(vnj-
                sm) + ur[0]*(vnj-sr)*(vnjd[nd]-smd[nd]) + pjd[nd];
            for (ii1 = 0; ii1 < NVARS; ++ii1)
                utempd[ii1][nd] = 0.0;
            utempd[0][nd] = ((urd[0][nd]*(sr-vnj)+ur[0]*(srd[nd]-vnjd[nd]))*(
                sr-sm)-ur[0]*(sr-vnj)*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        }
        pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;
        utemp[0] = ur[0]*(sr-vnj)/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[1][nd] = (((srd[nd]-vnjd[nd])*ur[1]+(sr-vnj)*urd[1][nd]+n[0
                ]*(pstard[nd]-pjd[nd]))*(sr-sm)-((sr-vnj)*ur[1]+(pstar-pj)*n[0
                ])*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        utemp[1] = ((sr-vnj)*ur[1]+(pstar-pj)*n[0])/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[2][nd] = (((srd[nd]-vnjd[nd])*ur[2]+(sr-vnj)*urd[2][nd]+n[1
                ]*(pstard[nd]-pjd[nd]))*(sr-sm)-((sr-vnj)*ur[2]+(pstar-pj)*n[1
                ])*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        utemp[2] = ((sr-vnj)*ur[2]+(pstar-pj)*n[1])/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[3][nd] = (((srd[nd]-vnjd[nd])*ur[3]+(sr-vnj)*urd[3][nd]-pjd
                [nd]*vnj-pj*vnjd[nd]+pstard[nd]*sm+pstar*smd[nd])*(sr-sm)-((sr
                -vnj)*ur[3]-pj*vnj+pstar*sm)*(srd[nd]-smd[nd]))/((sr-sm)*(sr-
                sm));
        utemp[3] = ((sr-vnj)*ur[3]-pj*vnj+pstar*sm)/(sr-sm);
        for (int ivar = 0; ivar < NVARS; ++ivar) {
            for (nd = 0; nd < NVARS; ++nd)
                fluxd[ivar*NVARS+nd] = fluxd[ivar*NVARS+nd] + srd[nd]*(utemp[ivar]-ur[
                    ivar]) + sr*(utempd[ivar][nd]-urd[ivar][nd]);
            flux[ivar] += sr*(utemp[ivar]-ur[ivar]);
        }
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = vnjd[nd]*ur[0] + vnj*urd[0][nd];
        flux[0] = vnj*ur[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = vnjd[nd]*ur[1] + vnj*urd[1][nd] + n[0]*pjd[nd];
        flux[1] = vnj*ur[1] + pj*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = vnjd[nd]*ur[2] + vnj*urd[2][nd] + n[1]*pjd[nd];
        flux[2] = vnj*ur[2] + pj*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = vnjd[nd]*(ur[3]+pj) + vnj*(urd[3][nd]+pjd[nd]);
        flux[3] = vnj*(ur[3]+pj);
    }
}

