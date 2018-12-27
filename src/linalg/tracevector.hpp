/** \file
 * \brief Classes for handling data associated with all the faces in  (the "trace of") a mesh
 * \author Aditya Kashi
 * \date 2018-12
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FVENS_TRACE_VECTOR_H
#define FVENS_TRACE_VECTOR_H

#include <vector>
#include "mesh/amesh2dh.hpp"

namespace fvens {

/// A vector belonging to the trace space of a discontinuous function defined on a mesh
/** Essentially a pair of "left" and "right" vectors defined on the trace of the mesh.
 * By trace of a function over the domain, we mean its restriction to the set of all faces of the mesh.
 * The faces are ordered in [the same way](@ref FaceIterators) as in the mesh object.
 * 
 * \warning Even though this class is templated on the scalar type, it is only for future use.
 *   Currently, only the fvens::a_real type works.
 */
template <typename scalar, int nvars>
class L2TraceVector
{
public:
	/// Sets the mesh and sets up communication pattern
	/** \param mesh The mesh over which the vector is defined. Must have its face structure available.
	 */
	L2TraceVector(const UMesh2dh<scalar>& mesh);

	/// Access to the "left" vector over all faces in the local subdomain
	scalar *getLocalArrayLeft();
	/// Access to the "right" vector over all faces in the local subdomain
	scalar *getLocalArrayRight();
	/// Immutable access to the "left" vector over all faces in the local subdomain
	const scalar *getLocalArrayLeft() const;
	/// Immutable access to the "right" vector over all faces in the local subdomain
	const scalar *getLocalArrayRight() const;

	/// Begin update the "right" vector from the appropriate neighbouring subdomains
	void updateSharedFacesBegin();
	/// Finish update the "right" vector from the appropriate neighbouring subdomains
	void updateSharedFacesEnd();

protected:
	/// The mesh over whose trace the vectors are defined
	const UMesh2dh<scalar>& m;
	/// Storage for left vector
	std::vector<scalar> left;
	/// Storage for right vector
	std::vector<scalar> right;

	/// MPI ranks of neighbouring subdomains of this subdomain
	std::vector<int> nbdranks;
	/// List of connectivity face indices associated with each of the neighbouring subdomains
	/** For each neighbouring subdomain, gives the connface index associated with each face.
	 */
	std::vector<std::vector<a_int>> sharedfaces;

	/// Connectivity face indices associated with shared faces as seen by the neighbouring subdomain
	/** This gives the connface index (of this subdomain) associated with each face of
	 *  the *other* subdomain shared with this one. So the size is same as \ref sharedfaces.
	 *  This is necessary because the other subdomain has a different ordering
	 *  of the shared connectivity faces.
	 */
	std::vector<std::vector<a_int>> sharedfacesother;

	/// Indices into the received buffer corresponding to the connectivity faces of this subdomain
	/** For each connectivity face in this subdomain, we store, in order,
	 *  - Index into nbdranks of the neighbouring subdomain which shares this face
	 *  NO: - Index (modulo nvars) into the buffer that will be received from that subdomain.
	 */
	amat::Array2d<a_int> sharedfacemap;

	/// Updates communication pattern and also updates the size of the vector
	/** Computes data needed for message passing. Should be called if the mesh topology changes.
	 * As currently implemented, this will cause all data to be lost, including vector entries.
	 */
	void update_comm_pattern();
};

}
#endif
