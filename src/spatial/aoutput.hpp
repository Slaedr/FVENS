/** @file aoutput.hpp
 * @brief A collection of subroutines to write mesh data to various kinds of output formats
 */

#ifndef AOUTPUT_H
#define AOUTPUT_H 1

#include "aspatial.hpp"

namespace fvens {

/// Interface for output to files
template <short nvars>
class Output
{
public:
	Output(const Spatial<a_real,nvars> *const fv);

	/// Exports data for the entire domain
	/** \param[in] u The field variables
	 * \param[in] volfile The name of the file to be written.
	 */
	virtual void exportVolumeData(const MVector<a_real>& u, const std::string volfile) const = 0;

	/// Exports data on surfaces
	/** \param[in] u The multi-vector containing field variables.
	 * \param[in] wbcm A list of `wall' boundary face markers for which output is needed.
	 * \param[in] obcm A list of `other' boundary face markers at which some other output is needed.
	 * \param[in] basename The base name for the files that will be written.
	 */
	virtual void exportSurfaceData(const MVector<a_real>& u, const std::vector<int> wbcm,
			const std::vector<int> obcm, const std::string basename) const = 0;

protected:
	const Spatial<a_real,nvars> *const space;
	const UMesh2dh<a_real> *const m;
};

/// Output for flow simulations
/** \todo We need a function that only returns the tuple of values but does not require an output
 * argument.
 */
class FlowOutput : public Output<NVARS>
{
public:
	/// Sets required data
	/** \param[in] angleOfAttack The angle of attack in radians
	 */
	FlowOutput(const FlowFV_base *const fv,
			const IdealGasPhysics<a_real> *const physics, const a_real angleOfAttack);
	
	/** For each cell, writes out the coordinates of the cell-centre,
	 * density, velocities, pressure, temperature and Mach number.
	 */
	void exportVolumeData(const MVector<a_real>& u, const std::string volfile) const;

	/// Export surface data
	/** We compute pressure and skin-friction coefficients for wall boundaries, and
	 * normalized x- and y-velocities along other boundaries.
	 * We also compute lift and drag for wall boundaries.
	 * \sa Output::exportSurfaceData
	 */
	void exportSurfaceData(const MVector<a_real>& u, const std::vector<int> wbcm,
			const std::vector<int> obcm, const std::string basename) const;

protected:
	//using Output<NVARS>::space;
	const FlowFV_base *const space;
	using Output<NVARS>::m;
	const IdealGasPhysics<a_real> *const phy;
	const a_real av[NDIM];				///< Unit vector in the direction of freestream flow
};

/** \brief Writes multiple scalar data sets and one vector data set, 
 * all cell-centered data, to a file in VTU format.
 * 
 * If either x or y is a 0x0 matrix, it is ignored.
 * \param fname is the output vtu file name
 */
void writeScalarsVectorToVtu_CellData(std::string fname, const UMesh2dh<a_real>& m, 
		const amat::Array2d<double>& x, std::string scaname[], 
		const amat::Array2d<double>& y, std::string vecname);

/// Writes nodal data to VTU file
void writeScalarsVectorToVtu_PointData(std::string fname, const UMesh2dh<a_real>& m, 
		const amat::Array2d<double>& x, std::string scaname[], 
		const amat::Array2d<double>& y, std::string vecname);

/// Writes a hybrid mesh in VTU format.
/** VTK does not have a 9-node quadrilateral, so we ignore the cell-centered note for output.
 */
void writeMeshToVtu(std::string fname, UMesh2dh<a_real>& m);

}
#endif
