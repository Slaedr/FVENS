/** \file
 * \brief Native implementations of and interfaces to some mesh orderings
 * \author Aditya Kashi
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

#include <vector>
#include <utility>
#include "utilities/adolcutils.hpp"
#include "meshordering.hpp"
#include "ameshutils.hpp"

namespace fvens {

using amat::Array2d;

template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m, const a_real threshold)
{
	const LineConfig lc = findLines(m, threshold);

	// Create the permutation vector using the lines
	std::vector<a_int> ordering(m.gnelem());
	a_int k = 0;
	for(size_t iline = 0; iline < lc.lines.size(); iline++)
	{
		for(a_int i = 0; i < static_cast<a_int>(lc.lines[iline].size()); i++) {
			ordering[k] = lc.lines[iline][i];
			k++;
		}
	}

	for(a_int iel = 0; iel < m.gnelem(); iel++)
		if(lc.celline[iel] == -1)
		{
			ordering[k] = iel;
			k++;
		}

	m.reorder_cells(&ordering[0]);
}

struct LocalAnisotropies
{
	/// Measure of local anisotropy for each cell for each neighbor, ordered decreasing
	Array2d<a_real> aniso;
	/// Local face index ordered according to \ref aniso
	Array2d<EIndex> faceIdx;
	/// Number of real neighbors for each cell
	std::vector<int> nRealNbrs;
};

/// Returns the edge weights about each cell, ordered by decreasing weight
template <typename scalar>
LocalAnisotropies computeWeights(const UMesh2dh<scalar>& m)
{
	LocalAnisotropies la;
	la.aniso.resize(m.gnelem(), m.gmaxnfael());
	la.faceIdx.resize(m.gnelem(), m.gmaxnfael());
	for(a_int i = 0; i < m.gnelem(); i++)
		for(int j = 0; j < m.gmaxnfael(); j++)
			la.faceIdx(i,j) = -1;
	la.nRealNbrs.resize(m.gnelem());

	Array2d<scalar> ccentres(m.gnelem(),NDIM);
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		kernel_computeCellCentreAoS(m, iel, &ccentres(iel,0));
	}

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		std::vector<std::pair<a_real,EIndex>> elaniso;
		elaniso.reserve(m.gnfael(iel));
		a_real minw = 1e20;

		for(EIndex j = 0; j < m.gnfael(iel); j++)
		{
			const a_int jel = m.gesuel(iel,j);
			// Skip ghost neighbors across boundary faces
			if(jel >= m.gnelem())
				continue;

			std::pair<a_real,EIndex> nbrwt;

			a_real dist = 0;
			for(int idim = 0; idim < NDIM; idim++)
				dist += std::pow(getvalue(ccentres(iel,idim)-ccentres(jel,idim)),2);
			nbrwt.first = 1.0/std::sqrt(dist);

			if(nbrwt.first < minw)
				minw = nbrwt.first;

			nbrwt.second = j;
			elaniso.push_back(nbrwt);
		}

		for(size_t j = 0; j < elaniso.size(); j++)
			elaniso[j].first /= minw;

		// sort by *decreasing* weight
		std::sort(elaniso.begin(), elaniso.end(),
		          [](std::pair<a_real,EIndex> a, std::pair<a_real,EIndex> b) {
			          return a.first > b.first;
		          });

		la.nRealNbrs[iel] = static_cast<int>(elaniso.size());
		for(EIndex j = 0; j < la.nRealNbrs[iel]; j++) {
			la.aniso(iel,j) = elaniso[j].first;
			la.faceIdx(iel,j) = elaniso[j].second;
		}
	}

	return la;
}

template <typename scalar>
LineConfig findLines(const UMesh2dh<scalar>& m, const a_real threshold)
{
	LineConfig lc;
	const LocalAnisotropies la = computeWeights(m);

	lc.celline.assign(m.gnelem(), -1);

	// Try to build a line starting at each boundary cell
	for(a_int iface = m.gPhyBFaceStart(); iface < m.gPhyBFaceEnd(); iface++)
	{
		std::vector<a_int> linelems;
		const a_int belem = m.gintfac(iface,0);
		if(lc.celline[belem] >= 0) {
			printf("  lineReorder: A boundary cell is already part of a line.\n");
			fflush(stdout);
			continue;
		}

		bool endoftheline = false;
		a_int curelem = belem;

		while(!endoftheline)
		{
			if(la.aniso(curelem,0) > threshold) {
				linelems.push_back(curelem);
				lc.celline[curelem] = static_cast<int>(lc.lines.size());
			}
			else
				break;

			endoftheline = true;

			for(EIndex j = 0; j < la.nRealNbrs[curelem]; j++)
			{
				const a_int nbrelem = m.gesuel(curelem, la.faceIdx(curelem,j));

				if(lc.celline[nbrelem]==-1 && la.aniso(curelem,j) > threshold) {
					curelem = nbrelem;
					endoftheline = false;
					break;
				}
			}
		}

		if(linelems.size() > 1)
			lc.lines.push_back(linelems);
		else if(linelems.size() == 1) {
			lc.celline[linelems[0]] = -1;
		}
	}

	printf(" lineReorder: Found %ld lines.\n", lc.lines.size());

	return lc;
}

template void lineReorder(UMesh2dh<a_real>& m, const a_real threshold);

}
