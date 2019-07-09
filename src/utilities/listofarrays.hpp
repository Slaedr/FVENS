/** \file
 * \brief Rudimentary abstraction for a certain implementation of lists of arrays
 */

#ifndef FVENS_LISTOFARRAYS_H
#define FVENS_LISTOFARRAYS_H

#include <array>
#include <vector>
#include "aconstants.hpp"

namespace fvens {

/// A list of arrays implemented as one long contiguous array
template <typename T>
struct ListOfArrays
{
	/// Storage for entries of all arrays in the list of arrays
	std::vector<T> store;
	/// Position integers which point into \ref store at locations where arrays begin
	std::vector<fint> ptrs;
};

/// A set of lists of arrays sharing the same topology determined by one array of pointers
template <typename T, int nlists>
struct ListsOfArrays
{
	/// A set of stores for a certain number of lists of arrays
	std::array<std::vector<T>,nlists> store;
	/// Posision integers which point into each \ref store at locations where arrays begin
	std::vector<fint> ptrs;
};

}
#endif
