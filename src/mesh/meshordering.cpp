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

#include <utility>
#include "meshordering.hpp"
#include "ameshutils.hpp"

namespace fvens {

/// Returns the edge weights about each cell, ordered by decreasing weight
template <typename scalar> static
std::pair<amat::Array2d<a_real>,amat::Array2d<EIndex>> computeWeights(const UMesh2dh<scalar>& m);

template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m, const double threshold)
{
}

template <typename scalar>
std::pair<amat::Array2d<a_real>,amat::Array2d<EIndex>> computeWeights(const UMesh2dh<scalar>& m)
{
	amat::Array2d<a_real> aniso(m.gnelem(), m.gmaxnfael());
	amat::Array2d<EIndex> faceIdx(m.gnelem(), m.gmaxnfael());

	amat::Array2d<scalar> ccentres(m.gnelem(),NDIM);
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		kernel_computeCellCentreAoS(m, iel, &ccentres(iel,0));
	}

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		std::vector<std::pair<a_real,EIndex>> elaniso(m.gnfael(iel));
		a_real avgw = 0;
		for(EIndex j = 0; j < m.gnfael(iel); j++)
		{
			const a_int jel = m.gelemface(iel,j);
			a_real dist = 0;
			for(int idim = 0; idim < NDIM; idim++)
				dist += std::pow(getvalue(ccentres(iel,idim)-ccentres(jel,idim)),2);

			elaniso[j].first = 1.0/std::sqrt(dist);
			avgw += elaniso[j].first;

			elaniso[j].second = j;
		}

		avgw /= m.gnfael(iel);
		for(EIndex j = 0; j < m.gnfael(iel); j++)
			elaniso[j].first /= avgw;

		// sort by *decreasing* weight
		std::sort(elaniso.begin(), elaniso.end(),
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

template void lineReorder(UMesh2dh<a_real>& m, const double threshold);

}
