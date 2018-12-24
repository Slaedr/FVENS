/** \file aoptionparser.cpp
 * \brief Parse options from different sources.
 * \author Aditya Kashi
 * \date 2017-10
 */

#include "aoptionparser.hpp"
#include "aerrorhandling.hpp"
#include "mpiutils.hpp"
#include <iostream>
#include <cstdlib>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>
#include <petscsys.h>

namespace fvens {

namespace po = boost::program_options;

po::variables_map parse_cmd_options(const int argc, const char *const argv[],
                                    po::options_description& desc)
{
	desc.add_options()
		("help", "Help message")
		("mesh_file", po::value<std::string>(),
		 "Mesh file to solve the problem on; overrides the corresponding option in the control file");

	po::variables_map cmdvarmap;
	po::parsed_options parsedopts =
		po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
	po::store(parsedopts, cmdvarmap);
	po::notify(cmdvarmap);

	return cmdvarmap;
}

bool parsePetscCmd_isDefined(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool flg = PETSC_FALSE;
	ierr = PetscOptionsHasName(NULL, NULL, optionname.c_str(), &flg);
	petsc_throw(ierr, "Could not determine whether PETSc option was defined!");
	if(flg == PETSC_TRUE)
		return true;
	else
		return false;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
int parsePetscCmd_int(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	int output = 0;
	ierr = PetscOptionsGetInt(NULL, NULL, optionname.c_str(), &output, &set);
	petsc_throw(ierr, std::string("Could not get int ")+ optionname);
	fvens_throw(!set, std::string("Int ") + optionname + std::string(" not set"));
	return output;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
PetscReal parseOptionalPetscCmd_real(const std::string optionname, const PetscReal defval)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	PetscReal output = 0;
	ierr = PetscOptionsGetReal(NULL, NULL, optionname.c_str(), &output, &set);
	petsc_throw(ierr, std::string("Could not get real ")+ optionname);
	if(!set) {
		std::cout << "PETSc cmd option " << optionname << " not set; using default.\n";
		output = defval;
	}
	return output;
}

bool parsePetscCmd_bool(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	PetscBool output = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL, NULL, optionname.c_str(), &output, &set);
	petsc_throw(ierr, std::string("Could not get bool ")+ optionname);
	fvens_throw(!set, std::string("Bool ") + optionname + std::string(" not set"));
	return (bool)output;
}


std::string parsePetscCmd_string(const std::string optionname, const size_t p_strlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	char* tt = new char[p_strlen+1];
	ierr = PetscOptionsGetString(NULL, NULL, optionname.data(), tt, p_strlen, &set);
	petsc_throw(ierr, std::string("Could not get string ") + std::string(optionname));
	fvens_throw(!set, std::string("String ") + optionname + std::string(" not set"));
	const std::string stropt = tt;
	delete [] tt;
	return stropt;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
std::vector<int> parsePetscCmd_intArray(const std::string optionname, const int maxlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	std::vector<int> arr(maxlen);
	int len = maxlen;

	ierr = PetscOptionsGetIntArray(NULL, NULL, optionname.c_str(), &arr[0], &len, &set);
	arr.resize(len);

	petsc_throw(ierr, std::string("Could not get array ") + std::string(optionname));
	fvens_throw(!set, std::string("Array ") + optionname + std::string(" not set"));
	return arr;
}

std::vector<int> parseOptionalPetscCmd_intArray(const std::string optionname, const int maxlen)
{
	StatusCode ierr = 0;
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);
	PetscBool set = PETSC_FALSE;
	std::vector<int> arr(maxlen);
	int len = maxlen;

	ierr = PetscOptionsGetIntArray(NULL, NULL, optionname.c_str(), &arr[0], &len, &set);
	arr.resize(len);

	petsc_throw(ierr, std::string("Could not get array ") + std::string(optionname));
	if(!set) {
		if(mpirank == 0)
			std::cout << "Array " << optionname << " not set.\n";
		arr.resize(0);
	}
	return arr;
}

}
