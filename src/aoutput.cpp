/** @file aoutput.cpp
 * @brief Implementation of subroutines to write mesh data to various kinds of output formats
 */

#include "aoutput.hpp"

namespace acfd {

template <short nvars>
Output<nvars>::Output(const UMesh2dh *const mesh, const Spatial<nvars> *const fv)
	: m(mesh), space(fv)
{ }

FlowOutput::FlowOutput(const UMesh2dh *const mesh, const Spatial<NVARS> *const fv,
		const IdealGasPhysics *const physics, const a_real aoa)
	: Output<NVARS>(mesh, fv), phy(physics), av{std::cos(aoa) ,std::sin(aoa)}
{
}

void FlowOutput::exportVolumeData(const MVector& u, std::string volfile) const
{
	// TODO
}

void FlowOutput::exportSurfaceData(const MVector& u, const std::vector<int> wbcm, 
		std::vector<int> obcm, std::string basename) const
{
	// Get conserved variables' gradients
	MVector grad[NDIM];
	for(int j = 0; j < NDIM; j++)
		grad[j].resize(m->gnelem(), NVARS);

	space->getGradients(u, grad);

	// get number of faces in wall boundary and other boundary
	
	std::vector<int> nwbfaces(wbcm.size(),0), nobfaces(obcm.size(),0);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		for(int im = 0; im < static_cast<int>(wbcm.size()); im++)
			if(m->ggallfa(iface,3) == wbcm[im]) nwbfaces[im]++;
		for(int im = 0; im < static_cast<int>(obcm.size()); im++)
			if(m->ggallfa(iface,3) == obcm[im]) nobfaces[im]++;
	}
	
	a_real pinf = phy->getFreestreamPressure();

	// unit vector normal to the free-stream flow direction
	a_real flownormal[NDIM]; flownormal[0] = -av[1]; flownormal[1] = av[0];

	// Iterate over wall boundary markers
	for(int im=0; im < static_cast<int>(wbcm.size()); im++)
	{
		std::string fname = basename+"-surf_w"+std::to_string(wbcm[im])+".dat";
		std::ofstream fout(fname);
		
		Matrix<a_real,Dynamic,Dynamic> output(nwbfaces[im], 2+NDIM);

		a_int facecoun = 0;			// face iteration counter for this boundary marker
		a_real totallen = 0;		// total area of the surface with this boundary marker
		a_real Cdf=0, Cdp=0, Cl=0;

		fout << "x \t y \t Cp  \t Cf \n";

		// iterate over faces having this boundary marker
		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			if(m->ggallfa(iface,3) == wbcm[im])
			{
				a_int lelem = m->gintfac(iface,0);
				a_real n[NDIM];
				for(int j = 0; j < NDIM; j++)
					n[j] = m->ggallfa(iface,j);
				const a_real len = m->ggallfa(iface,2);
				totallen += len;

				// coords of face center
				a_real ijp[NDIM];
				ijp[0] = m->gintfac(iface,2);
				ijp[1] = m->gintfac(iface,3);
				a_real coord[NDIM];
				for(int j = 0; j < NDIM; j++) 
				{
					coord[j] = 0;
					for(int inofa = 0; inofa < m->gnnofa(); inofa++)
						coord[j] += m->gcoords(ijp[inofa],j);
					coord[j] /= m->gnnofa();
					
					output(facecoun,j) = coord[j];
				}

				/** Pressure coefficient: 
				 * \f$ C_p = (p-p_\infty)/(\frac12 rho_\infty * v_\infty^2) \f$
				 * = p* - p_inf* where *'s indicate non-dimensional values.
				 * We note that p_inf* = 1/(gamma Minf^2).
				 */
				output(facecoun, NDIM) = (phy->getPressureFromConserved(&u(lelem,0)) - pinf)/2.0;

				/** Skin friction coefficient \f% C_f = \tau_w / (\frac12 \rho v_\infty^2) \f$.
				 * 
				 * We can define \f$ \tau_w \f$, the wall shear stress, as
				 * \f$ \tau_w = (\mathbf{\Tau} \hat{\mathbf{n}}).\hat{\mathbf{t}} \f$
				 * where \f$ \mathbf{\Tau} \f$ is the viscous stress tensor, 
				 * \f$ \hat{\mathbf{n}} \f$ is the unit normal to the face and 
				 * \f$ \hat{\mathbf{t}} \f$ is a consistent unit tangent to the face.
				 * 
				 * Note that because of our non-dimensionalization,
				 * \f$ C_f = \hat{\mu} \tau_w / 2 \f$.
				 *
				 * But we could maybe also use, for incompressible flow,
				 * \f$ \tau_w = \mu \nabla\mathbf{u} \hat{\mathbf{n}} . \hat{\mathbf{t}} \f$.
				 */

				// non-dim viscosity / Re_inf
				a_real muhat = phy->getViscosityCoeffFromConserved(&u(lelem,0));
				a_real lhat = -2.0/3.0 * muhat;

				// velocity gradient tensor
				a_real gradu[NDIM][NDIM];
				gradu[0][0] = (grad[0](lelem,1)*u(lelem,0)-u(lelem,1)*grad[0](lelem,0))
				                / (u(lelem,0)*u(lelem,0));
				gradu[0][1] = (grad[1](lelem,1)*u(lelem,0)-u(lelem,1)*grad[1](lelem,0))
				                / (u(lelem,0)*u(lelem,0));
				gradu[1][0] = (grad[0](lelem,2)*u(lelem,0)-u(lelem,2)*grad[0](lelem,0))
				                / (u(lelem,0)*u(lelem,0));
				gradu[1][1] = (grad[1](lelem,2)*u(lelem,0)-u(lelem,2)*grad[1](lelem,0))
				                / (u(lelem,0)*u(lelem,0));

				// stress tensor
				a_real tau[NDIM][NDIM];
				a_real divu = gradu[0][0] + gradu[1][1];
				tau[0][0] = lhat*divu + 2.0*muhat*gradu[0][0];
				tau[0][1] = muhat*(gradu[0][1]+gradu[1][0]);
				tau[1][0] = tau[0][1];
				tau[1][1] = lhat*divu + 2.0*muhat*gradu[1][1];

				a_real tauw = -tau[0][0]*n[0]*n[1] - tau[0][1]*n[1]*n[1]
					+ tau[1][0]*n[0]*n[0] + tau[1][1]*n[0]*n[1];

				output(iface, NDIM+1) = 2.0*tauw;

				// add contributions to Cdp, Cdf and Cl
				
				// face normal dot free-stream direction
				const a_real ndotf = n[0]*av[0]+n[1]*av[1];
				// face normal dot "up" direction perpendicular to free stream
				const a_real ndotnf = n[0]*flownormal[0]+n[1]*flownormal[1];
				// face tangent dot free-stream direction
				const a_real tdotf = -n[1]*av[0]+n[0]*av[1];

				Cdp += output(facecoun,NDIM)*ndotf*len;
				Cdf += output(facecoun,NDIM+1)*tdotf*len;
				Cl += output(facecoun,NDIM)*ndotnf*len;

				facecoun++;
			}
		}

		// Normalize drag and lift by reference area
		Cdp /= totallen; Cdf /= totallen; Cl /= totallen;

		// write out the output

		for(a_int i = 0; i < output.rows(); i++)
		{
			for(int j = 0; j < output.cols(); j++)
				fout << "  " << output(i,j);
			fout << '\n';
		}

		fout << "# Cl      Cdp      Cdf\n";
		fout << "# " << Cl << "  " << Cdp << "  " << Cdf << '\n';

		fout.close();
	}

	// Iterate over `other' boundary markers and compute normalized velocities
	
	for(int im=0; im < static_cast<int>(obcm.size()); im++)
	{
		std::string fname = basename+"-surf_o"+std::to_string(obcm[im])+".dat";
		std::ofstream fout(fname);
		
		Matrix<a_real,Dynamic,Dynamic> output(nobfaces[im], 2+NDIM);
		a_int facecoun = 0;

		fout << "#   x         y          u           v\n";

		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			if(m->ggallfa(iface,3) == obcm[im])
			{
				a_int lelem = m->gintfac(iface,0);
				/*a_real n[NDIM];
				for(int j = 0; j < NDIM; j++)
					n[j] = m->ggallfa(iface,j);
				const a_real len = m->ggallfa(iface,2);*/

				// coords of face center
				a_real ijp[NDIM];
				ijp[0] = m->gintfac(iface,2);
				ijp[1] = m->gintfac(iface,3);
				a_real coord[NDIM];
				for(int j = 0; j < NDIM; j++) 
				{
					coord[j] = 0;
					for(int inofa = 0; inofa < m->gnnofa(); inofa++)
						coord[j] += m->gcoords(ijp[inofa],j);
					coord[j] /= m->gnnofa();
					
					output(facecoun,j) = coord[j];
				}

				output(facecoun,NDIM) =  u(lelem,1)/u(lelem,0);
				output(facecoun,NDIM+1)= u(lelem,2)/u(lelem,0);

				facecoun++;
			}
		}
		
		// write out the output

		for(a_int i = 0; i < output.rows(); i++)
		{
			for(int j = 0; j < output.cols(); j++)
				fout << "  " << output(i,j);
			fout << '\n';
		}

		fout.close();
	}
}

void writeScalarsVectorToVtu_CellData(std::string fname, const acfd::UMesh2dh& m, 
		const amat::Array2d<double>& x, std::string scaname[], 
		const amat::Array2d<double>& y, std::string vecname)
{
	int elemcode;
	std::cout << "aoutput: Writing vtu output to " << fname << "\n";
	std::ofstream out(fname);

	int nscalars = x.cols();

	out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
	out << "<UnstructuredGrid>\n";
	out << "\t<Piece NumberOfPoints=\"" << m.gnpoin() << "\" NumberOfCells=\"" << m.gnelem() 
		<< "\">\n";
	
	if(x.msize()>0 || y.msize()>0) {
		out << "\t\t<CellData ";
		if(x.msize() > 0)
			out << "Scalars=\"" << scaname[0] << "\" ";
		if(y.msize() > 0)
			out << "Vectors=\"" << vecname << "\"";
		out << ">\n";
	}
	
	//enter cell scalar data if available
	if(x.msize() > 0) {
		for(int in = 0; in < nscalars; in++)
		{
			out << "\t\t\t<DataArray type=\"Float64\" Name=\"" << scaname[in] 
				<< "\" Format=\"ascii\">\n";
			for(int i = 0; i < m.gnelem(); i++)
				out << "\t\t\t\t" << x.get(i,in) << '\n';
			out << "\t\t\t</DataArray>\n";
		}
		//cout << "aoutput: Scalars written.\n";
	}

	//enter vector cell data if available
	if(y.msize() > 0) {
		out << "\t\t\t<DataArray type=\"Float64\" Name=\"" << vecname 
			<< "\" NumberOfComponents=\"3\" Format=\"ascii\">\n";
		for(int i = 0; i < m.gnelem(); i++)
		{
			out << "\t\t\t\t";
			for(int idim = 0; idim < y.cols(); idim++)
				out << y.get(i,idim) << " ";
			if(y.cols() == 2)
				out << "0.0 ";
			out << '\n';
		}
		out << "\t\t\t</DataArray>\n";
	}
	if(x.msize() > 0 || y.msize() > 0)
		out << "\t\t</CellData>\n";

	//enter points
	out << "\t\t<Points>\n";
	out << "\t\t<DataArray type=\"Float64\" NumberOfComponents=\"3\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnpoin(); i++)
	{
		out << "\t\t\t";
		for(int idim = 0; idim < NDIM; idim++)
			out << m.gcoords(i,idim) << " ";
		if(NDIM == 2)
			out << "0.0 ";
		out << '\n';
	}
	out << "\t\t</DataArray>\n";
	out << "\t\t</Points>\n";

	//enter cells
	out << "\t\t<Cells>\n";
	out << "\t\t\t<DataArray type=\"UInt32\" Name=\"connectivity\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++) 
	{
		out << "\t\t\t\t"; 
		
		elemcode = 5;
		if(m.gnnode(i) == 4)
			elemcode = 9;
		else if(m.gnnode(i) == 6)
			elemcode = 22;
		else if(m.gnnode(i) == 8)
			elemcode = 23;
		else if(m.gnnode(i) == 9)
			elemcode = 28;
		
		for(int inode = 0; inode < m.gnnode(i); inode++)	
			out << m.ginpoel(i,inode) << " ";
		out << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" Format=\"ascii\">\n";
	int totalcells = 0;
	for(int i = 0; i < m.gnelem(); i++) {
		totalcells += m.gnnode(i);
		out << "\t\t\t\t" << totalcells << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t\t<DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++) {
		elemcode = 5;
		if(m.gnnode(i) == 4)
			elemcode = 9;
		else if(m.gnnode(i) == 6)
			elemcode = 22;
		else if(m.gnnode(i) == 8)
			elemcode = 23;
		else if(m.gnnode(i) == 9)
			elemcode = 28;
		out << "\t\t\t\t" << elemcode << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t</Cells>\n";

	//finish upper
	out << "\t</Piece>\n";
	out << "</UnstructuredGrid>\n";
	out << "</VTKFile>";
	out.close();
	std::cout << "Vtu file written.\n";
}

void writeScalarsVectorToVtu_PointData(std::string fname, const acfd::UMesh2dh& m, 
		const amat::Array2d<double>& x, std::string scaname[], 
		const amat::Array2d<double>& y, std::string vecname)
{
	int elemcode;
	std::cout << "aoutput: Writing vtu output to " << fname << "\n";
	std::ofstream out(fname);

	int nscalars = x.cols();

	out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
	out << "<UnstructuredGrid>\n";
	out << "\t<Piece NumberOfPoints=\"" << m.gnpoin() << "\" NumberOfCells=\"" << m.gnelem() 
		<< "\">\n";
	
	if(x.msize()>0 || y.msize()>0) {
		out << "\t\t<PointData ";
		if(x.msize() > 0)
			out << "Scalars=\"" << scaname[0] << "\" ";
		if(y.msize() > 0)
			out << "Vectors=\"" << vecname << "\"";
		out << ">\n";
	}
	
	//enter cell scalar data if available
	if(x.msize() > 0) {
		for(int in = 0; in < nscalars; in++)
		{
			out << "\t\t\t<DataArray type=\"Float64\" Name=\"" << scaname[in] 
				<< "\" Format=\"ascii\">\n";
			for(int i = 0; i < m.gnpoin(); i++)
				out << "\t\t\t\t" << x.get(i,in) << '\n';
			out << "\t\t\t</DataArray>\n";
		}
		//cout << "aoutput: Scalars written.\n";
	}

	//enter vector cell data if available
	if(y.msize() > 0) {
		out << "\t\t\t<DataArray type=\"Float64\" Name=\"" << vecname 
			<< "\" NumberOfComponents=\"3\" Format=\"ascii\">\n";
		for(int i = 0; i < m.gnpoin(); i++)
		{
			out << "\t\t\t\t";
			for(int idim = 0; idim < y.cols(); idim++)
				out << y.get(i,idim) << " ";
			if(y.cols() == 2)
				out << "0.0 ";
			out << '\n';
		}
		out << "\t\t\t</DataArray>\n";
	}
	if(x.msize() > 0 || y.msize() > 0)
		out << "\t\t</PointData>\n";

	//enter points
	out << "\t\t<Points>\n";
	out << "\t\t<DataArray type=\"Float64\" NumberOfComponents=\"3\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnpoin(); i++)
	{
		out << "\t\t\t";
		for(int idim = 0; idim < NDIM; idim++)
			out << m.gcoords(i,idim) << " ";
		if(m.gndim() == 2)
			out << "0.0 ";
		out << '\n';
	}
	out << "\t\t</DataArray>\n";
	out << "\t\t</Points>\n";

	//enter cells
	out << "\t\t<Cells>\n";
	out << "\t\t\t<DataArray type=\"UInt32\" Name=\"connectivity\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++) 
	{
		out << "\t\t\t\t"; 
		
		elemcode = 5;
		if(m.gnnode(i) == 4)
			elemcode = 9;
		else if(m.gnnode(i) == 6)
			elemcode = 22;
		else if(m.gnnode(i) == 8)
			elemcode = 23;
		else if(m.gnnode(i) == 9)
			elemcode = 28;
		
		for(int inode = 0; inode < m.gnnode(i); inode++)	
			out << m.ginpoel(i,inode) << " ";
		out << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" Format=\"ascii\">\n";
	int totalcells = 0;
	for(int i = 0; i < m.gnelem(); i++) {
		totalcells += m.gnnode(i);
		out << "\t\t\t\t" << totalcells << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t\t<DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++) {
		elemcode = 5;
		if(m.gnnode(i) == 4)
			elemcode = 9;
		else if(m.gnnode(i) == 6)
			elemcode = 22;
		else if(m.gnnode(i) == 8)
			elemcode = 23;
		else if(m.gnnode(i) == 9)
			elemcode = 28;
		out << "\t\t\t\t" << elemcode << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t</Cells>\n";

	//finish upper
	out << "\t</Piece>\n";
	out << "</UnstructuredGrid>\n";
	out << "</VTKFile>";
	out.close();
	std::cout << "Vtu file written.\n";
}


/// Writes a hybrid mesh in VTU format.
/** VTK does not have a 9-node quadrilateral, so we ignore the cell-centered note for output.
 */
void writeMeshToVtu(std::string fname, acfd::UMesh2dh& m)
{
	std::cout << "Writing vtu output...\n";
	std::ofstream out(fname);

	out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
	out << "<UnstructuredGrid>\n";
	out << "\t<Piece NumberOfPoints=\"" << m.gnpoin() << "\" NumberOfCells=\"" << m.gnelem() 
		<< "\">\n";

	//enter points
	out << "\t\t<Points>\n";
	out << "\t\t<DataArray type=\"Float64\" NumberOfComponents=\"3\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnpoin(); i++)
		out << "\t\t\t" << m.gcoords(i,0) << " " << m.gcoords(i,1) << " " << 0.0 << '\n';
	out << "\t\t</DataArray>\n";
	out << "\t\t</Points>\n";

	//enter cells
	out << "\t\t<Cells>\n";
	out << "\t\t\t<DataArray type=\"UInt32\" Name=\"connectivity\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++)
	{
		out << "\t\t\t\t";
		for(int j = 0; j < m.gnnode(i); j++)
			out << m.ginpoel(i,j) << " ";
		out << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++)
		out << "\t\t\t\t" << m.gnnode(i)*(i+1) << '\n';
	out << "\t\t\t</DataArray>\n";
	out << "\t\t\t<DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n";
	for(int i = 0; i < m.gnelem(); i++)
	{
		int elemcode = 5;
		if(m.gnnode(i) == 4)
			elemcode = 9;
		else if(m.gnnode(i) == 6)
			elemcode = 22;
		else if(m.gnnode(i) == 8)
			elemcode = 23;
		else if(m.gnnode(i) == 9)
			elemcode = 23;
		out << "\t\t\t\t" << elemcode << '\n';
	}
	out << "\t\t\t</DataArray>\n";
	out << "\t\t</Cells>\n";

	//finish upper
	out << "\t</Piece>\n";
	out << "</UnstructuredGrid>\n";
	out << "</VTKFile>";
	out.close();
	std::cout << "Vtu file written.\n";
}

}
