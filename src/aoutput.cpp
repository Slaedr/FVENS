/** @file aoutput.cpp
 * @brief Implementation of subroutines to write mesh data to various kinds of output formats
 */

#include "aoutput.hpp"

namespace acfd {

template <short nvars>
Output<nvars>::Output(const UMesh2dh *const mesh, const Spatial<nvars> *const fv)
	: m(mesh), space(fv)
{ }

FlowOutput::FlowOutput(const UMesh2dh *const mesh, const Spatial<NVARS> *const fv)
	: Output<NVARS>(mesh, fv)
{
	//
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
	
	int nwbfaces = 0, nobfaces = 0;
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		for(int im = 0; im < static_cast<int>(wbcm.size()); im++)
			if(m->ggallfa(iface,3) == wbcm[im]) nwbfaces++;
		for(int im = 0; im < static_cast<int>(obcm.size()); im++)
			if(m->ggallfa(iface,3) == obcm[im]) nobfaces++;
	}

	// Iterate over wall boundary markers
	for(int im=0; im < static_cast<int>(wbcm.size()); im++)
	{
		int bsf = wbcm[im];

		std::string fname = basename+"-surf_"+".dat";
		Matrix<a_real,Dynamic,Dynamic> output(nwbfaces, 3+NDIM);

		// iterate over faces
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
