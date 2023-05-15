
#ifndef FVENS_GHOSTVECCOMM_H
#define FVENS_GHOSTVECCOMM_H

#include <petscvec.h>
#include "linalg/tracevector.hpp"
#include "utilities/aerrorhandling.hpp"

namespace fvens {

/// Communication direction for ghosted vectors
enum VecCommMode {
                  DOMAIN_TO_GHOST,           ///< Communicate domain values to neighbouring ghost cells
                  GHOST_TO_DOMAIN            ///< Communicate ghost cell values to neighbouring domain
};

/// Handles communication for a ghosted vector
template <int bs>
class GhostedBlockVecComm
{
public:
	/// Set the mesh and setup communication pattern
	GhostedBlockVecComm(const UMesh<PetscReal,NDIM> *const mesh);
	/// Set the vector to communicate
	void setVec(Vec vv);
	/// Set communication mode and insertion mode
	/** Insertion mode is taken from PETSc - ADD_VALUES or INSERT_VALUES
	 */
	void setModes(const VecCommMode vmode, const InsertMode inmode);
	/// Begin sending values
	void vecUpdateBegin();
	/// Finish receiving values
	void vecUpdateEnd();

protected:
	const UMesh<PetscReal,NDIM> *const msh;  ///< Mesh context

	/// Temporary storage used for communication
	/** TODO: Replace with more efficient class that only stores connectivity boundary values.
	 * Currently, ALL face values are stored, most of which are not used.
	 */
	L2TraceVector<PetscReal,bs> ltv;

	bool commUnderWay;   ///< Whether a communication is currently ongoind
	Vec vv;              ///< Pointer to vector currently being communcated
	VecCommMode cmode;   ///< Current communication direction
	InsertMode insmode;  ///< Current insertion mode
};

}
#endif
