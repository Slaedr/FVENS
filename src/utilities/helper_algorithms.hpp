/** \file
 * \brief Implementation of some discrete algorithms 
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

#ifndef FVENS_HELPER_ALGORITHMS_H
#define FVENS_HELPER_ALGORITHMS_H

#include <algorithm>
#include "helper_algorithms.hpp"

namespace fvens {

template <typename index>
inline void inclusive_scan(std::vector<index>& v)
{
	// serial
	for(size_t i = 1; i < v.size(); i++)
		v[i] += v[i-1];
}

template <typename index, typename allocator>
inline void inclusive_scan(std::vector<index,allocator>& v)
{
	// serial
	for(size_t i = 1; i < v.size(); i++)
		v[i] += v[i-1];
}

template <typename index>
inline std::vector<index> inclusive_scan(const std::vector<index>& v)
{
	std::vector<index> scanned(v);
	inclusive_scan(scanned);
	return scanned;
}

}
#endif
