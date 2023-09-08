/** \file aerrorhandling.cpp
 * \brief Implementation file for error handling
 * \author Aditya Kashi
 */

#include <iostream>

#include <petscsys.h>

#include "aerrorhandling.hpp"

namespace fvens {
	
MPI_exception::MPI_exception(const std::string& msg) 
	: std::runtime_error(std::string("MPI error: ")+msg)
{ }

MPI_exception::MPI_exception(const char *const msg) 
	: std::runtime_error(std::string("MPI error: ") + std::string(msg))
{ }

namespace {

std::string get_petsc_error_message(const int error_code)
{
	const char *error_msg_s;
	char *context_s;
	int ierr = PetscErrorMessage(error_code, &error_msg_s, &context_s);
	if(ierr) {
		std::cout << "Could not get PETSc error message!" << std::endl;
	}
	std::string error_msg(error_msg_s), context(context_s);
	return error_msg + ": " + context;
}

}
	
Petsc_exception::Petsc_exception(const int ierr, const std::string& msg)
	: std::runtime_error(std::string("PETSc error: ") + get_petsc_error_message(ierr) + msg)
{ }

Petsc_exception::Petsc_exception(const int ierr, const char *const msg)
	: std::runtime_error(std::string("PETSc error: ") + get_petsc_error_message(ierr)
						 + std::string(msg))
{ }

Numerical_error::Numerical_error(const std::string& msg) 
	: std::logic_error(msg)
{ }

Tolerance_error::Tolerance_error(const std::string& msg) 
	: Numerical_error(msg)
{ }

InputNotGivenError::InputNotGivenError(const std::string& msg) 
	: std::runtime_error(std::string("Input not given: ")+msg)
{ }

UnsupportedOptionError::UnsupportedOptionError(const std::string& msg)
	: std::runtime_error(std::string("Unsupported option: ")+msg)
{ }

void open_file_toRead(const std::string file, std::ifstream& fin)
{
	fin.open(file);
	if(!fin) {
		std::cout << "! Could not open file "<< file <<" !\n";
		std::abort();
	}
}

void open_file_toWrite(const std::string file, std::ofstream& fout)
{
	fout.open(file);
	if(!fout) {
		throw std::runtime_error("Could not open file " + file);
	}
}

}
