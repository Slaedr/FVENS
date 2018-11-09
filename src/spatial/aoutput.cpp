/** @file aoutput.cpp
 * @brief Implementation of subroutines to write mesh data to various kinds of output formats
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "aoutput.hpp"
#include "utilities/aoptionparser.hpp"

namespace fvens {

template <short nvars>
Output<nvars>::Output(const Spatial<a_real,nvars> *const fv)
	: space(fv), m(space->mesh())
{ }

FlowOutput::FlowOutput(const FlowFV_base<a_real> *const fv,
                       const IdealGasPhysics<a_real> *const physics, const a_real aoa)
	: Output<NVARS>(fv), space(fv), phy(physics), angleOfAttack{aoa},
	av{std::cos(aoa) ,std::sin(aoa)}
{
}

a_real FlowOutput::compute_entropy_cell(const Vec uvec) const
{
	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr);

	std::array<a_real,NVARS> uinf = phy->compute_freestream_state(angleOfAttack);

	a_real sinf = phy->getEntropyFromConserved(&uinf[0]);

	amat::Array2d<a_real> s_err(m->gnelem(),1);
	a_real error = 0;
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		s_err(iel) = (phy->getEntropyFromConserved(&uarr[iel*NVARS]) - sinf) / sinf;
		error += s_err(iel)*s_err(iel)*m->garea(iel);
	}
	error = sqrt(error);

	a_real h = 1.0/sqrt(m->gnelem());
 
	std::cout << "FlowOutput: log mesh size and log entropy:   " << log10(h) << "  " 
	          << std::setprecision(10) << log10(error) << std::endl;

	ierr = VecRestoreArrayRead(uvec, &uarr);
	(void)ierr;
	return error;
}

StatusCode FlowOutput::postprocess_cell(const Vec uvec, 
                                        amat::Array2d<a_real>& scalars, 
                                        amat::Array2d<a_real>& velocities) const
{
	std::cout << "FlowFV: postprocess_cell(): Creating output arrays...\n";
	scalars.resize(m->gnelem(), 3);
	velocities.resize(m->gnelem(), NDIM);
	
	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector<a_real>> u(uarr, m->gnelem(), NVARS);

	for(a_int iel = 0; iel < m->gnelem(); iel++) {
		scalars(iel,0) = u(iel,0);
	}

	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		velocities(iel,0) = u(iel,1)/u(iel,0);
		velocities(iel,1) = u(iel,2)/u(iel,0);
		a_real vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
		scalars(iel,2) = phy->getPressureFromConserved(&uarr[iel*NVARS]);
		a_real c = phy->getSoundSpeedFromConserved(&uarr[iel*NVARS]);
		scalars(iel,1) = sqrt(vmag2)/c;
	}
	compute_entropy_cell(uvec);
	
	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	std::cout << "FlowFV: postprocess_cell(): Done.\n";
	return ierr;
}

StatusCode FlowOutput::postprocess_point(const Vec uvec,
                                         amat::Array2d<a_real>& scalars,
                                         amat::Array2d<a_real>& velocities) const
{
	std::cout << "FlowFV: postprocess_point(): Creating output arrays...\n";
	scalars.resize(m->gnpoin(),4);
	velocities.resize(m->gnpoin(),NDIM);

	amat::Array2d<a_real> areasum(m->gnpoin(),1);
	amat::Array2d<a_real> up(m->gnpoin(), NVARS);
	up.zeros();
	areasum.zeros();

	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector<a_real>> u(uarr, m->gnelem(), NVARS);

	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int inode = 0; inode < m->gnnode(ielem); inode++)
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				up(m->ginpoel(ielem,inode),ivar) += u(ielem,ivar)*m->garea(ielem);
				areasum(m->ginpoel(ielem,inode)) += m->garea(ielem);
			}
	}

	for(a_int ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(int ivar = 0; ivar < NVARS; ivar++)
			up(ipoin,ivar) /= areasum(ipoin);
	
	for(a_int ipoin = 0; ipoin < m->gnpoin(); ipoin++)
	{
		scalars(ipoin,0) = up(ipoin,0);

		for(int idim = 0; idim < NDIM; idim++)
			velocities(ipoin,idim) = up(ipoin,idim+1)/up(ipoin,0);
		const a_real vmag2 = dimDotProduct(&velocities(ipoin,0),&velocities(ipoin,0));

		scalars(ipoin,2) = phy->getPressureFromConserved(&up(ipoin,0));
		a_real c = phy->getSoundSpeedFromConserved(&up(ipoin,0));
		scalars(ipoin,1) = sqrt(vmag2)/c;
		scalars(ipoin,3) = phy->getTemperatureFromConserved(&up(ipoin,0));
	}

	compute_entropy_cell(uvec);

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	std::cout << "FlowFV: postprocess_point(): Done.\n";
	return ierr;
}

void FlowOutput::exportVolumeData(const MVector<a_real>& u, std::string volfile) const
{
	std::ofstream fout;
	open_file_toWrite(volfile+"-vol.out", fout);
	fout << "#   x    y    rho     u      v      p      T      M \n";

	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		const a_real T = phy->getTemperatureFromConserved(&u(iel,0));
		const a_real c = phy->getSoundSpeedFromConserved(&u(iel,0));
		const a_real p = phy->getPressureFromConserved(&u(iel,0));
		a_real vmag = std::sqrt(u(iel,1)/u(iel,0)*u(iel,1)/u(iel,0)
				+u(iel,2)/u(iel,0)*u(iel,2)/u(iel,0));

		a_real rc[NDIM] = {0.0,0.0};
		for(int ino = 0; ino < m->gnnode(iel); ino++)
			for(int j = 0; j < NDIM; j++)
				rc[j] += m->gcoords(m->ginpoel(iel,ino),j);
		for(int j = 0; j < NDIM; j++)
			rc[j] /= m->gnnode(iel);

		fout << rc[0] << " " << rc[1] << " " << u(iel,0) << " " << u(iel,1)/u(iel,0) << " "
			<< u(iel,2)/u(iel,0) << " " << p << " " << T << " " << vmag/c << '\n';
	}

	fout.close();
}

/** \todo Use values at the face to compute drag, lift etc. rather than cell-centred data.
 */
void FlowOutput::exportSurfaceData(const MVector<a_real>& u, const std::vector<int> wbcm, 
		std::vector<int> obcm, const std::string basename) const
{
	// Get conserved variables' gradients
	GradArray<a_real,NVARS> grad;
	grad.resize(m->gnelem());

	space->getGradients(u, grad);

	// get number of faces in wall boundary and other boundary
	
	std::vector<int> nwbfaces(wbcm.size(),0), nobfaces(obcm.size(),0);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		for(int im = 0; im < static_cast<int>(wbcm.size()); im++)
			if(m->gintfacbtags(iface,0) == wbcm[im]) nwbfaces[im]++;
		for(int im = 0; im < static_cast<int>(obcm.size()); im++)
			if(m->gintfacbtags(iface,0) == obcm[im]) nobfaces[im]++;
	}

	// Iterate over wall boundary markers
	for(int im=0; im < static_cast<int>(wbcm.size()); im++)
	{
		std::string fname = basename+"-surf_w"+std::to_string(wbcm[im])+".out";
		std::ofstream fout; 
		open_file_toWrite(fname, fout);
		
		MVector<a_real> output(nwbfaces[im], 2+NDIM);

		//a_int facecoun = 0;			// face iteration counter for this boundary marker
		//a_real totallen = 0;		// total area of the surface with this boundary marker
		a_real Cdf=0, Cdp=0, Cl=0;

		fout << "#  x \t y \t Cp  \t Cf \n";

		// iterate over faces having this boundary marker
		std::tie(Cl, Cdp, Cdf) = space->computeSurfaceData(u, grad, wbcm[im], output);

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

		std::cout << "FlowOutput: CL = " << Cl << "   CDp = " << Cdp << "    CDf = " << Cdf
			<< std::endl;
	}

	// Iterate over `other' boundary markers and compute normalized velocities
	
	for(int im=0; im < static_cast<int>(obcm.size()); im++)
	{
		std::string fname = basename+"-surf_o"+std::to_string(obcm[im])+".out";
		std::ofstream fout;
		open_file_toWrite(fname, fout);
		
		Matrix<a_real,Dynamic,Dynamic> output(nobfaces[im], 2+NDIM);
		a_int facecoun = 0;

		fout << "#   x         y          u           v\n";

		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			if(m->gintfacbtags(iface,0) == obcm[im])
			{
				a_int lelem = m->gintfac(iface,0);
				/*a_real n[NDIM];
				for(int j = 0; j < NDIM; j++)
					n[j] = m->gfacemetric(iface,j);
				const a_real len = m->gfacemetric(iface,2);*/

				// coords of face center
				a_int ijp[NDIM];
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

void writeScalarsVectorToVtu_CellData(std::string fname, const fvens::UMesh2dh<a_real>& m, 
                                      const amat::Array2d<double>& x, std::string scaname[], 
                                      const amat::Array2d<double>& y, std::string vecname)
{
	int elemcode;
	std::cout << "aoutput: Writing vtu output to " << fname << "\n";
	std::ofstream out;
	open_file_toWrite(fname, out);
	
	out << std::setprecision(10);

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

void writeScalarsVectorToVtu_PointData(std::string fname, const fvens::UMesh2dh<a_real>& m, 
                                       const amat::Array2d<double>& x, std::string scaname[], 
                                       const amat::Array2d<double>& y, std::string vecname)
{
	int elemcode;
	std::cout << "aoutput: Writing vtu output to " << fname << "\n";
	std::ofstream out;
	open_file_toWrite(fname, out);
	
	out << std::setprecision(10);

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


/// Writes a hybrid mesh in VTU format.
/** VTK does not have a 9-node quadrilateral, so we ignore the cell-centered note for output.
 */
void writeMeshToVtu(std::string fname, fvens::UMesh2dh<a_real>& m)
{
	std::cout << "Writing vtu output...\n";
	std::ofstream out(fname);
	
	out << std::setprecision(10);

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

void writeConvergenceHistoryHeader(std::ostream& outf)
{
	using std::setw;
	outf << '#' << setw(6) << "NStep" << setw(16) << "Log10 rel resi" << setw(16) << "Log10 abs resi"
	     << setw(12) << "Tot.Wtime" << setw(12) << "Lin.Wtime" << setw(12) << "Lin.iters"
	     << setw(10) << "CFL" << '\n';
	outf << "#------------------------------------------------------------------------------------\n";
}

void writeStepToConvergenceHistory(const SteadyStepMonitor s, std::ostream& outf)
{
	using std::setw;
	outf << std::setprecision(6);
	outf << setw(7) << s.step << setw(16) << std::log10(s.rmsres) << setw(16)
	     << std::log10(s.absrmsres);
	outf << std::setprecision(4);
	outf << setw(12) << s.odewalltime << setw(12) << s.linwalltime
	     << setw(12) << s.linits << setw(10) << s.cfl << '\n';
	outf << std::flush;
}

}
