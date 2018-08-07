/** \file abctypemap.hpp
 * \brief Declaration of bi-directional map between BC types and BC type strings
 * \author Aditya Kashi
 */

#ifndef FVENS_BCTYPEMAP_H
#define FVENS_BCTYPEMAP_H

#include <boost/bimap.hpp>
#include "abctypes.hpp"

namespace fvens {

/// Global bi-directional map for getting from the type enum to type string and vice-versa
extern boost::bimap<BCType, std::string> bcTypeMap;

/// Call this function somewhere to initialize \ref bcTypeMap
void setBCTypeMap();

}
#endif
