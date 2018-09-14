/** \file aoptionparser.hpp
 * \brief Some helper functions for parsing options from different sources.
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef AOPTIONPARSER_H
#define AOPTIONPARSER_H

#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include "aconstants.hpp"

namespace fvens {

/// Opens a file for reading but aborts in case of an error
void open_file_toRead(const std::string file, std::ifstream& fin);

/// Opens a file for writing but aborts in case of an error
void open_file_toWrite(const std::string file, std::ofstream& fout);

/// Parses command line parameters into a map
boost::program_options::variables_map
parse_cmd_options(const int argc, const char *const argv[],
                  boost::program_options::options_description& desc);

/// Extracts an integer corresponding to the argument from the default PETSc options database 
/** Throws an exception if the option was not set or if it could not be extracted.
 * \param optionname The name of the option to get the value of; needs to include the preceding '-'
 */
int parsePetscCmd_int(const std::string optionname);

/// Optionally extracts a real corresponding to the argument from the default PETSc options database 
/** Throws an exception if the function to read the option fails, but not if it succeeds and reports
 * that the option was not set.
 * \param optionname Name of the option to be extracted
 * \param defval The default value to be assigned in case the option was not passed
 */
PetscReal parseOptionalPetscCmd_real(const std::string optionname, const PetscReal defval);

/// Extracts a boolean corresponding to the argument from the default PETSc options database 
/** Throws an exception if the option was not set or if it could not be extracted.
 * \param optionname The name of the option to get the value of; needs to include the preceding '-'
 */
bool parsePetscCmd_bool(const std::string optionname);

/// Extracts a string corresponding to the argument from the default PETSc options database 
/** Throws a string exception if the option was not set or if it could not be extracted.
 * \param optionname The name of the option to get the value of; needs to include the preceding '-'
 * \param len The max number of characters expected in the string value
 */
std::string parsePetscCmd_string(const std::string optionname, const size_t len);

/// Extracts the arguments of an int array option from the default PETSc options database
/** \param maxlen Maximum number of entries expected in the array
 * \return The vector of array entries; its size is the number of elements read, no more
 */
std::vector<int> parsePetscCmd_intArray(const std::string optionname, const int maxlen);

/// Extracts the arguments of an int array option from the default PETSc options database
/** Does not throw if the requested option was not found; just returns an empty vector in that case. 
 * \param maxlen Maximum number of entries expected in the array
 */
std::vector<int> parseOptionalPetscCmd_intArray(const std::string optionname, const int maxlen);

}

#endif
