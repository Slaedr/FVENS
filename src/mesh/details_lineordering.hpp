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
	std::vector<std::vector<fint>> lines;
	/// For each cell, stores the line number it belongs to, if applicable, otherwise stores -1
	std::vector<int> celline;
};

/// Finds lines in the mesh
template <typename scalar>
LineConfig findLines(const UMesh<scalar,2>& m, const freal threshold);

/** Below, when we say 'point', we mean 'cell that does not belong to any line', unless otherwise stated
 */
struct GraphVertex
{
	bool isline;             ///< Whether this entry is a line or a point
	fint idx;               ///< Index of line or point in the list of lines or points resp.
};

/// Details needed 
struct GraphVertices
{
	/// List of graph vertices
	std::vector<GraphVertex> gverts;

	/// Contiguous list of 'points' (cells not belonging to any line); holds cell indices of points.
	std::vector<fint> pointList;

	/** Holds, for each cell, its point index or -1 depending on whether it's a point
	 * or in a line respectively.
	 */
	std::vector<fint> cellsToPtsMap;

	/// The lines configuration for the mesh
	const LineConfig *lc;
};

/// Creates the list of line/point vertices from the mesh and its line configuration
template <typename scalar>
GraphVertices createLinePointGraphVertices(const UMesh<scalar,2>& m, const LineConfig& lc);

/// Creates a graph in which vertices represent lines and points (which are not in lines) in the mesh
template <typename scalar>
void createLinePointGraph(const UMesh<freal,2>& m, const GraphVertices& gv, Mat *const G);

/// Uses PETSc to get the ordering
std::vector<PetscInt> getPetscOrdering(Mat G, const char *const ordering);

/// Computes the cell ordering from the graph of lines and points
template <typename scalar>
std::vector<fint> getHybridLineOrdering(const UMesh<scalar,2>& m, const freal threshold,
                                         const char *const ordering);

}

#endif
