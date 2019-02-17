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

namespace fvens {

template <typename scalar> static
std::pair<amat::Array2d<scalar>,amat::Array2d<a_int>> computeWeights(const UMesh2dh<scalar>& m);

template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m, const double threshold)
{
}

template <typename scalar>
std::pair<amat::Array2d<scalar>,amat::Array2d<a_int>> computeWeights(const UMesh2dh<scalar>& m)
{
	amat::Array2d<scalar> weights(m.gnelem(),2);
	amat::Array2d<a_int> faceNbrIdx(m.gnelem(),2);

	return std::make_pair(weights,faceNbrIdx);
}

template void lineReorder(UMesh2dh<a_real>& m, const double threshold);

}
