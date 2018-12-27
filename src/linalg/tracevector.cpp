/** \file
 * \brief Implementation of trace vectors
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

#include <set>
#include "utilities/mpiutils.hpp"
#include "tracevector.hpp"

namespace fvens {

template <typename scalar, int nvars>
L2TraceVector<scalar,nvars>::L2TraceVector(const UMesh2dh<scalar>& mesh) : m{mesh}
{
	update_comm_pattern();
}

template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::update_comm_pattern()
{
	left.resize(m.gnaface());
	right.resize(m.gnaface());

	sharedfacemap.resize(m.gnConnFace(),1);

	std::set<int> nbds;
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		nbds.insert(m.gconnface(icface,2));
	}
	nbdranks.resize(nbds.size());
	std::copy(nbds.begin(), nbds.end(), nbdranks.begin());

	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		for(int i = 0; i < static_cast<int>(nbdranks.size()); i++)
			if(m.gconnface(icface,2) == nbdranks[i])
				sharedfacemap(icface,0) = i;
	}

	sharedfaces.resize(nbdranks.size());
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		sharedfaces[sharedfacemap(icface,0)].push_back(icface);
	}

	sharedfacesother.resize(nbdranks.size());
	for(size_t irank = 0; irank < nbdranks.size(); irank++) {
		sharedfacesother[irank].assign(sharedfaces[irank].size(), -1);
	}

	std::vector<std::vector<a_int>> nbdElemIndex(nbdranks.size());
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		nbdElemIndex[irank].resize(sharedfaces[irank].size());

	std::vector<MPI_Request> rrequests(nbdranks.size());

	// Send and receive element indices corresponding to shared faces
	{
		std::vector<MPI_Request> srequests(nbdranks.size());
		std::vector<std::vector<a_int>> myElemIndex(nbdranks.size());

		for(size_t irank = 0; irank < nbdranks.size(); irank++)
		{
			myElemIndex[irank].resize(sharedfaces[irank].size());
			for(size_t iface = 0; iface < sharedfaces[irank].size(); iface++)
				myElemIndex[irank][iface] = m.gconnface(sharedfaces[irank][iface],0);

			const int dest = nbdranks[irank];
			const int tag = irank;
			MPI_Isend(&myElemIndex[irank][0], myElemIndex[irank].size(), FVENS_MPI_INT, dest, tag,
			          MPI_COMM_WORLD, &srequests[irank]);
		}

		for(size_t irank = 0; irank < nbdranks.size(); irank++)
		{
			const int source = nbdranks[irank];
			const int tag = irank;
			MPI_Irecv(&nbdElemIndex[irank][0], nbdElemIndex[irank].size(), FVENS_MPI_INT, source, tag,
			          MPI_COMM_WORLD, &rrequests[irank]);
		}

		// Wait for sends to finish
		for(size_t irank = 0; irank < nbdranks.size(); irank++)
			MPI_Wait(&srequests[irank], MPI_STATUS_IGNORE);
	}

	// Wait for receives to finish
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		MPI_Wait(&rrequests[irank], MPI_STATUS_IGNORE);

	// set the mapping between connectivity faces and receive buffers
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		const int rankindex = sharedfacemap(icface,0);
		const a_int elem = m.gconnface(icface,3);

		for(a_int isface = 0; isface < static_cast<a_int>(sharedfaces[rankindex].size()); icface++)
		{
			if(nbdElemIndex[rankindex][isface] == elem) {
				//sharedfacemap(icface,1) = isface;
				sharedfacesother[nbdranks[rankindex]][isface] = icface;
				break;
			}
		}
	}

	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		for(size_t isface = 0; isface < sharedfacesother.size(); isface++)
			assert(sharedfacesother[irank][isface] != -1);
}

template <typename scalar, int nvars>
scalar *L2TraceVector<scalar,nvars>::getLocalArrayLeft()
{
	return &left[0];
}

template <typename scalar, int nvars>
scalar *L2TraceVector<scalar,nvars>::getLocalArrayRight()
{
	return &right[0];
}

template <typename scalar, int nvars>
const scalar *L2TraceVector<scalar,nvars>::getLocalArrayLeft() const
{
	return &left[0];
}

template <typename scalar, int nvars>
const scalar *L2TraceVector<scalar,nvars>::getLocalArrayRight() const
{
	return &right[0];
}

/** A more general implementation would use MPI_BYTE and manually serialize the array of scalars.
 * But the easier way would be to define templated MPI communication functions and use those below.
 */
template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::updateSharedFacesBegin()
{
	std::vector<MPI_Request> sreq(nbdranks.size());
	std::vector<std::vector<scalar>> buffers(nbdranks.size());

	const a_int fstart = m.gConnBFaceStart();
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
	{
		buffers[irank].resize(sharedfaces[irank].size()*nvars);
		for(size_t iface = 0; iface < sharedfaces[irank].size(); iface++)
		{
			for(int ivar = 0; ivar < nvars; ivar++)
				buffers[irank][iface*nvars+ivar]
					= left [(sharedfaces[irank][iface]-fstart)*nvars + ivar];
		}

		MPI_Isend(&buffers[irank][0], sharedfaces.size()*nvars, FVENS_MPI_REAL, nbdranks[irank], irank,
		          MPI_COMM_WORLD, &sreq[irank]);
	}

	std::vector<MPI_Status> status(nbdranks.size());
	MPI_Waitall(nbdranks.size(), &sreq[0], &status[0]);
}

template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::updateSharedFacesEnd()
{
	std::vector<MPI_Request> rreq(nbdranks.size());
	std::vector<std::vector<scalar>> buffers(nbdranks.size());

	for(size_t irank = 0; irank < nbdranks.size(); irank++)
	{
		buffers[irank].resize(sharedfaces[irank].size()*nvars);
		MPI_Irecv(&buffers[irank][0], sharedfaces.size()*nvars, FVENS_MPI_REAL, nbdranks[irank], irank,
		          MPI_COMM_WORLD, &rreq[irank]);
	}

	const a_int fstart = m.gConnBFaceStart();
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
	{
		MPI_Wait(&rreq[irank], MPI_STATUS_IGNORE);

		for(size_t isface = 0; isface < sharedfaces[irank].size(); isface++)
		{
			for(int ivar = 0; ivar < nvars; ivar++)
				right[(fstart+sharedfacesother[irank][isface])*nvars+ivar]
					= buffers[irank][isface*nvars+ivar];
		}
	}

}

template class L2TraceVector<a_real,1>;
template class L2TraceVector<a_real,NVARS>;

}
