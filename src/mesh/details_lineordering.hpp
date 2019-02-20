/** \file
 * \brief Some declarations required for implementing line orderings
 * \author Aditya Kashi
 */

#ifndef FVENS_DETAILS_LINE_ORDERING
#define FVENS_DETAILS_LINE_ORDERING

#include <petscmat.h>

namespace fvens {

/// Description of lines in the mesh
struct LineConfig {
	/// Indices of cells that make up lines
	std::vector<std::vector<a_int>> lines;
	/// For each cell, stores the line number it belongs to, if applicable, otherwise stores -1
	std::vector<int> celline;
};

/// Finds lines in the mesh
template <typename scalar>
LineConfig findLines(const UMesh2dh<scalar>& m, const a_real threshold);

/** Below, when we say 'point', we mean 'cell that does not belong to any line', unless otherwise stated
 */
struct GraphVertex
{
	bool isline;             ///< Whether this entry is a line or a point
	a_int idx;               ///< Index of line or point in the list of lines or points resp.
};

/// Details needed 
struct GraphVertices
{
	/// List of graph vertices
	std::vector<GraphVertex> gverts;

	/// Create list of points (cells not belonging to any line)
	/** pointList holds cell indices of 'points', while cellsToPtsMap holds, for each cell,
	 *  its point index or -1 depending on whether it's a point or in a line respectively.
	 */
	std::vector<a_int> pointList;
	std::vector<a_int> cellsToPtsMap;

	/// The lines configuration for the mesh
	const LineConfig *lc;
};

/// Creates the list of line/point vertices from the mesh and its line configuration
template <typename scalar>
GraphVertices createLinePointGraphVertices(const UMesh2dh<scalar>& m, const LineConfig& lc);

/// Creates a graph in which vertices represent lines and points (which are not in lines) in the mesh
template <typename scalar>
void createLinePointGraph(const UMesh2dh<a_real>& m, const GraphVertices& gv, Mat *const G);

/// Uses PETSc to get the ordering
std::vector<PetscInt> getPetscOrdering(Mat G, const char *const ordering);

/// Computes the cell ordering from the graph of lines and points
template <typename scalar>
std::vector<a_int> getHybridLineOrdering(const UMesh2dh<scalar>& m, const a_real threshold,
                                         const char *const ordering);

}

#endif
