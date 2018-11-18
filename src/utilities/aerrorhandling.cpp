/** \file aerrorhandling.cpp
 * \brief Implementation file for error handling
 * \author Aditya Kashi
 */

#include "aerrorhandling.hpp"

namespace fvens {
	
MPI_exception::MPI_exception(const std::string& msg) 
	: std::runtime_error(std::string("MPI error: ")+msg)
{ }

MPI_exception::MPI_exception(const char *const msg) 
	: std::runtime_error(std::string("MPI error: ") + std::string(msg))
{ }
	
Petsc_exception::Petsc_exception(const std::string& msg) 
	: std::runtime_error(std::string("PETSc error: ")+msg)
{ }

Petsc_exception::Petsc_exception(const char *const msg) 
	: std::runtime_error(std::string("PETSc error: ") + std::string(msg))
{ }

Numerical_error::Numerical_error(const std::string& msg) 
	: std::logic_error(msg)
{ }

Tolerance_error::Tolerance_error(const std::string& msg) 
	: Numerical_error(msg)
{ }

}
