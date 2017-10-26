/** \file autilities.hpp
 * \brief Some helper functions for mundane tasks
 */

#ifndef AUTILITIES_H
#define AUTILITIES_H

#include <fstream>

namespace acfd {

/// Opens a file for reading but aborts in case of an error
std::ifstream open_file_toRead(const std::string file);

/// Opens a file for writing but aborts in case of an error
std::ofstream open_file_toWrite(const std::string file);

}

#endif
