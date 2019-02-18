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

struct LineConfig {
	std::vector<std::vector<a_int>> lines;         ///< Cell indices of lines
	std::vector<bool> inaline;                     ///< Stores for each cell whether it's in any line
};

/// Returns the edge weights about each cell, ordered by decreasing weight
template <typename scalar> static
std::pair<Array2d<a_real>,Array2d<EIndex>> computeWeights(const UMesh2dh<scalar>& m);

/// Finds lines in the mesh
template <typename scalar> static
LineConfig findLines(const UMesh2dh<scalar>& m, const a_real threshold);

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
		if(!lc.inaline[iel])
		{
			ordering[k] = iel;
			k++;
		}

	m.reorder_cells(&ordering[0]);
}

template <typename scalar>
std::pair<Array2d<a_real>,Array2d<EIndex>> computeWeights(const UMesh2dh<scalar>& m)
{
	Array2d<a_real> aniso(m.gnelem(), m.gmaxnfael());
	Array2d<EIndex> faceIdx(m.gnelem(), m.gmaxnfael());

	Array2d<scalar> ccentres(m.gnelem(),NDIM);
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		kernel_computeCellCentreAoS(m, iel, &ccentres(iel,0));
	}

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		std::vector<std::pair<a_real,EIndex>> elaniso(m.gnfael(iel));
		a_real minw = 1e20;
		for(EIndex j = 0; j < m.gnfael(iel); j++)
		{
			const a_int jel = m.gelemface(iel,j);
			a_real dist = 0;
			for(int idim = 0; idim < NDIM; idim++)
				dist += std::pow(getvalue(ccentres(iel,idim)-ccentres(jel,idim)),2);

			elaniso[j].first = 1.0/std::sqrt(dist);
			if(elaniso[j].first < minw)
				minw = elaniso[j].first;

			elaniso[j].second = j;
		}

		for(EIndex j = 0; j < m.gnfael(iel); j++)
			elaniso[j].first /= minw;

		// sort by *decreasing* weight
		std::sort(elaniso.begin(), elaniso.begin()+m.gnfael(iel),
		          [](std::pair<a_real,EIndex> a, std::pair<a_real,EIndex> b) {
			          return a.first > b.first;
		          });

		for(EIndex j = 0; j < m.gnfael(iel); j++) {
			aniso(iel,j) = elaniso[j].first;
			faceIdx(iel,j) = elaniso[j].second;
		}
	}

	return std::make_pair(aniso,faceIdx);
}

template <typename scalar>
LineConfig findLines(const UMesh2dh<scalar>& m, const a_real threshold)
{
	LineConfig lc;
	const std::pair<Array2d<a_real>,Array2d<EIndex>> weights = computeWeights(m);

	lc.inaline.assign(m.gnelem(), false);

	// Try to build a line starting at each boundary cell
	for(a_int iface = m.gPhyBFaceStart(); iface < m.gPhyBFaceEnd(); iface++)
	{
		std::vector<a_int> linelems;
		const a_int belem = m.gintfac(iface,0);
		if(lc.inaline[belem]) {
			printf("  lineReorder: A boundary cell is already part of a line!\n");
			fflush(stdout);
			break;
		}

		bool endoftheline = false;
		a_int curelem = belem;

		while(!endoftheline)
		{
			if(weights.first(curelem,0) > threshold) {
				linelems.push_back(curelem);
				lc.inaline[curelem] = true;
			}
			else
				break;

			endoftheline = true;

			for(EIndex j = 0; j < m.gnfael(curelem); j++) {
				const a_int nbrelem = m.gelemface(curelem, weights.second(curelem,j));

				if(!lc.inaline[nbrelem] && weights.first(curelem,j) > threshold) {
					curelem = nbrelem;
					endoftheline = false;
					break;
				}
			}
		}

		if(linelems.size() > 1)
			lc.lines.push_back(linelems);
	}

	return lc;
}

template void lineReorder(UMesh2dh<a_real>& m, const a_real threshold);

}
