/** @file aoutput.cpp
 * @brief Implementation of subroutines to write mesh data to various kinds of output formats
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "aoutput.hpp"
#include "utilities/aerrorhandling.hpp"
#include "utilities/mpiutils.hpp"
#include "physics/aphysics_defs.hpp"

namespace fvens {

template <short nvars>
Output<nvars>::Output(const Spatial<freal,nvars> *const fv)
	: space(fv), m(space->mesh())
{ }

FlowOutput::FlowOutput(const FlowFV_base<freal> *const fv,
                       const IdealGasPhysics<freal> *const physics, const freal aoa)
	: Output<NVARS>(fv), space(fv), phy(physics), angleOfAttack{aoa},
	av{std::cos(aoa) ,std::sin(aoa)}
{
}

freal FlowOutput::compute_entropy_cell(const Vec uvec) const
{
	StatusCode ierr = 0;
	const int mpirank = get_mpi_rank(MPI_COMM_WORLD);

	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr);
	petsc_throw(ierr, "Could not get vec array!");

	std::array<freal,NVARS> uinf = phy->compute_freestream_state(angleOfAttack);

	const freal sinf = phy->getEntropyFromConserved(&uinf[0]);

	//amat::Array2d<freal> s_err(m->gnelem(),1);
	freal error = 0;
#pragma omp parallel for default(shared) reduction(+:error)
	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		const freal s_err = (phy->getEntropyFromConserved(&uarr[iel*NVARS]) - sinf) / sinf;
		error += s_err*s_err*m->garea(iel);
	}

	mpi_all_reduce<freal>(&error, 1, MPI_SUM, PETSC_COMM_WORLD);
	error = sqrt(error);

	const freal h = 1.0/sqrt(m->gnelem());

	if(mpirank == 0)
		std::cout << "FlowOutput: log mesh size and log entropy:   " << log10(h) << "  " 
		          << std::setprecision(10) << log10(error) << std::endl;

	ierr = VecRestoreArrayRead(uvec, &uarr);
	petsc_throw(ierr, "Could not restore vec!");
	return error;
}

StatusCode FlowOutput::postprocess_cell(const Vec uvec, 
                                        amat::Array2d<freal>& scalars, 
                                        amat::Array2d<freal>& velocities) const
{
	std::cout << "FlowFV: postprocess_cell(): Creating output arrays...\n";
	scalars.resize(m->gnelem(), 3);
	velocities.resize(m->gnelem(), NDIM);
	
	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector<freal>> u(uarr, m->gnelem(), NVARS);

	for(fint iel = 0; iel < m->gnelem(); iel++) {
		scalars(iel,0) = u(iel,0);
	}

	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		velocities(iel,0) = u(iel,1)/u(iel,0);
		velocities(iel,1) = u(iel,2)/u(iel,0);
		freal vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
		scalars(iel,2) = phy->getPressureFromConserved(&uarr[iel*NVARS]);
		freal c = phy->getSoundSpeedFromConserved(&uarr[iel*NVARS]);
		scalars(iel,1) = sqrt(vmag2)/c;
	}
	compute_entropy_cell(uvec);
	
	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	std::cout << "FlowFV: postprocess_cell(): Done.\n";
	return ierr;
}

StatusCode FlowOutput::postprocess_point(const Vec uvec,
                                         amat::Array2d<freal>& scalars,
                                         amat::Array2d<freal>& velocities) const
{
	std::cout << "FlowFV: postprocess_point(): Creating output arrays...\n";
	scalars.resize(m->gnpoin(),4);
	velocities.resize(m->gnpoin(),NDIM);

	amat::Array2d<freal> areasum(m->gnpoin(),1);
	amat::Array2d<freal> up(m->gnpoin(), NVARS);
	up.zeros();
	areasum.zeros();

	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector<freal>> u(uarr, m->gnelem(), NVARS);

	for(fint ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int inode = 0; inode < m->gnnode(ielem); inode++)
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				up(m->ginpoel(ielem,inode),ivar) += u(ielem,ivar)*m->garea(ielem);
				areasum(m->ginpoel(ielem,inode)) += m->garea(ielem);
			}
	}

	for(fint ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(int ivar = 0; ivar < NVARS; ivar++)
			up(ipoin,ivar) /= areasum(ipoin);
	
	for(fint ipoin = 0; ipoin < m->gnpoin(); ipoin++)
	{
		scalars(ipoin,0) = up(ipoin,0);

		for(int idim = 0; idim < NDIM; idim++)
			velocities(ipoin,idim) = up(ipoin,idim+1)/up(ipoin,0);
		const freal vmag2 = dimDotProduct(&velocities(ipoin,0),&velocities(ipoin,0));

		scalars(ipoin,2) = phy->getPressureFromConserved(&up(ipoin,0));
		freal c = phy->getSoundSpeedFromConserved(&up(ipoin,0));
		scalars(ipoin,1) = sqrt(vmag2)/c;
		scalars(ipoin,3) = phy->getTemperatureFromConserved(&up(ipoin,0));
	}

	compute_entropy_cell(uvec);

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	std::cout << "FlowFV: postprocess_point(): Done.\n";
	return ierr;
}

void FlowOutput::exportVolumeData(const MVector<freal>& u, std::string volfile) const
{
	std::ofstream fout;
	open_file_toWrite(volfile+"-vol.out", fout);
	fout << "#   x    y    rho     u      v      p      T      M \n";

	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		const freal T = phy->getTemperatureFromConserved(&u(iel,0));
		const freal c = phy->getSoundSpeedFromConserved(&u(iel,0));
		const freal p = phy->getPressureFromConserved(&u(iel,0));
		freal vmag = std::sqrt(u(iel,1)/u(iel,0)*u(iel,1)/u(iel,0)
				+u(iel,2)/u(iel,0)*u(iel,2)/u(iel,0));

		freal rc[NDIM] = {0.0,0.0};
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

/** \todo Generalize for MPI runs.
 * \todo Use values at the face to compute drag, lift etc. rather than cell-centred data.
 */
void FlowOutput::exportSurfaceData(const Vec u,
                                   const std::vector<int> wbcm, std::vector<int> obcm,
                                   const std::string basename                         ) const
{
	const int mpirank = get_mpi_rank(MPI_COMM_WORLD);

	// Get conserved variables' gradients
	std::vector<GradBlock_t<freal,NDIM,NVARS>> grad;
	grad.resize(m->gnelem());

	space->getGradients(u, &grad[0]);

	ConstVecHandler<freal> uh(u);
	const amat::Array2dView<freal> ua(uh.getArray(), m->gnelem(), NVARS);

	// get number of faces in wall boundary and other boundary
	
	std::vector<int> nwbfaces(wbcm.size(),0), nobfaces(obcm.size(),0);

	for(fint iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		for(int im = 0; im < static_cast<int>(wbcm.size()); im++)
			if(m->gbtags(iface,0) == wbcm[im]) nwbfaces[im]++;
		for(int im = 0; im < static_cast<int>(obcm.size()); im++)
			if(m->gbtags(iface,0) == obcm[im]) nobfaces[im]++;
	}

	// Iterate over wall boundary markers
	for(int im=0; im < static_cast<int>(wbcm.size()); im++)
	{
		std::string fname = basename+"-surf_w"+std::to_string(wbcm[im])+".out";
		std::ofstream fout; 
		open_file_toWrite(fname, fout);
		
		MVector<freal> output(nwbfaces[im], 2+NDIM);

		freal Cdf=0, Cdp=0, Cl=0;

		fout << "#  x \t y \t Cp  \t Cf \n";

		// iterate over faces having this boundary marker
		std::tie(Cl, Cdp, Cdf) = space->computeSurfaceData(ua, &grad[0], wbcm[im], output);

		// write out the output

		for(fint i = 0; i < output.rows(); i++)
		{
			for(int j = 0; j < output.cols(); j++)
				fout << "  " << output(i,j);
			fout << '\n';
		}

		fout << "# Cl      Cdp      Cdf\n";
		fout << "# " << Cl << "  " << Cdp << "  " << Cdf << '\n';

		fout.close();

		if(mpirank == 0)
			std::cout << "FlowOutput: CL = " << Cl << "   CDp = " << Cdp << "    CDf = " << Cdf
			          << std::endl;
	}

	// Iterate over `other' boundary markers and compute normalized velocities
	
	for(int im=0; im < static_cast<int>(obcm.size()); im++)
	{
		std::string fname = basename+"-surf_o"+std::to_string(obcm[im])+".out";
		std::ofstream fout;
		open_file_toWrite(fname, fout);

		MVector<freal> output(nobfaces[im], 2+NDIM);
		fint facecoun = 0;

		fout << "#   x         y          u           v\n";

		for(fint iface = 0; iface < m->gnbface(); iface++)
		{
			if(m->gbtags(iface,0) == obcm[im])
			{
				fint lelem = m->gintfac(iface,0);
				/*freal n[NDIM];
				for(int j = 0; j < NDIM; j++)
					n[j] = m->gfacemetric(iface,j);
				const freal len = m->gfacemetric(iface,2);*/

				// coords of face center
				fint ijp[NDIM];
				ijp[0] = m->gintfac(iface,2);
				ijp[1] = m->gintfac(iface,3);
				freal coord[NDIM];
				for(int j = 0; j < NDIM; j++) 
				{
					coord[j] = 0;
					for(int inofa = 0; inofa < m->gnnofa(iface); inofa++)
						coord[j] += m->gcoords(ijp[inofa],j);
					coord[j] /= m->gnnofa(iface);
					
					output(facecoun,j) = coord[j];
				}

				output(facecoun,NDIM) =  ua(lelem,1)/ua(lelem,0);
				output(facecoun,NDIM+1)= ua(lelem,2)/ua(lelem,0);

				facecoun++;
			}
		}
		
		// write out the output

		for(fint i = 0; i < output.rows(); i++)
		{
			for(int j = 0; j < output.cols(); j++)
				fout << "  " << output(i,j);
			fout << '\n';
		}

		fout.close();
	}
}

void writeScalarsVectorToVtu_CellData(std::string fname, const fvens::UMesh<freal,NDIM>& m, 
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

void writeScalarsVectorToVtu_PointData(std::string fname, const fvens::UMesh<freal,NDIM>& m, 
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
void writeMeshToVtu(std::string fname, const fvens::UMesh<freal,NDIM>& m)
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
	outf << '#' << setw(6) << "NStep" << setw(14) << "Log rel resi" << setw(14) << "Log abs resi"
	     << setw(12) << "Tot.Wtime" << setw(12) << "Lin.Wtime" << setw(12) << "Lin.iters"
	     << setw(10) << "CFL" << '\n';
	outf << "#----------------------------------------------------------------------------------\n";
}

void writeStepToConvergenceHistory(const SteadyStepMonitor s, std::ostream& outf)
{
	using std::setw;
	outf << std::setprecision(6);
	outf << setw(7) << s.step << setw(14) << std::log10(s.rmsres) << setw(14)
	     << std::log10(s.absrmsres);
	outf << std::setprecision(4);
	outf << setw(12) << s.odewalltime << setw(12) << s.linwalltime
	     << setw(12) << s.linits << setw(10) << s.cfl << '\n';
	outf << std::flush;
}

}
