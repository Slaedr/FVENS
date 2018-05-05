/** \file aerrorhandling.cpp
 * \brief Implementation file for error handling
 * \author Aditya Kashi
 */

#include "aerrorhandling.hpp"

namespace acfd {
	
Petsc_exception::Petsc_exception(const std::string& msg) 
	: std::runtime_error(std::string("PETSc error: ")+msg)
{ }

Petsc_exception::Petsc_exception(const char *const msg) 
	: std::runtime_error(std::string("PETSc error: ") + std::string(msg))
{ }

Numerical_error::Numerical_error(const std::string& msg) 
	: std::logic_error(msg)
{ }

}
