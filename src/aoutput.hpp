/** @file aoutput.hpp
 * @brief A collection of subroutines to write mesh data to various kinds of output formats
 */

#ifndef __AOUTPUT_H

#ifndef __AMATRIX_H
#include <amatrix.hpp>
#endif

#ifndef __AMESH2DHYBRID_H
#include <amesh2dh.hpp>
#endif

#ifndef _GLIBCXX_FSTREAM
#include <fstream>
#endif

#ifndef _GLIBCXX_STRING
#include <string>
#endif

#define __AOUTPUT_H 1

/** Writes multiple scalar data sets and one vector data set, all cell-centered data, to a file in VTU format.
 * If either x or y is a 0x0 matrix, it is ignored.
 * \param fname is the output vtu file name
 */
void writeScalarsVectorToVtu_CellData(std::string fname, const acfd::UMesh2dh& m, const amat::Matrix<double>& x, std::string scaname[], const amat::Matrix<double>& y, std::string vecname);

/// Writes nodal data to VTU file
void writeScalarsVectorToVtu_PointData(std::string fname, const acfd::UMesh2dh& m, const amat::Matrix<double>& x, std::string scaname[], const amat::Matrix<double>& y, std::string vecname);

/// Writes a hybrid mesh in VTU format.
/** VTK does not have a 9-node quadrilateral, so we ignore the cell-centered note for output.
 */
void writeMeshToVtu(std::string fname, acfd::UMesh2dh& m);


#endif
