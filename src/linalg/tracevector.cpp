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
L2TraceVector<scalar,nvars>::update_comm_pattern()
{
	left.resize(m.gnaface());
	right.resize(m.gnaface());

	sharedfacemap.resize(m.gnConnFace(),2);

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

		for(a_int isface = 0; isface < nbdElemIndex[rankindex]; icface++)
		{
			if(nbdElemIndex[rankindex][isface] == elem) {
				sharedfacemap(icface,1) = isface;
				break;
			}
		}
	}
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

template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::updateSharedFaces()
{
}

template class L2TraceVector<a_real,1>;
template class L2TraceVector<a_real,NVARS>;

}
