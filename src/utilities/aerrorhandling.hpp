/** \file aerrorhandling.hpp
 * \brief Exceptions and other error handling routines for FVENS
 * \author Aditya Kashi
 */

#ifndef AERRORHANDLING_H
#define AERRORHANDLING_H

#include <stdexcept>
#include <string>

namespace fvens {

/// Exception to throw on errors related to numerics
class Numerical_error : public std::logic_error
{
public:
	/// Construct a numerical error with a message
	Numerical_error(const std::string& msg);
};
	
/// An exception to throw for errors from PETSc; takes a custom message
class Petsc_exception : public std::runtime_error
{
public:
	Petsc_exception(const std::string& msg);
	Petsc_exception(const char *const msg);
};

/// Throw an error from an error code
/** \param ierr An expression which, if true, triggers the exception
 * \param str A short string message describing the error
 */
inline void fvens_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw std::runtime_error(str);
}

/// Throw an error from an error code related to PETSc
/** \param ierr An expression which, if true, triggers the exception
 * \param str A short string message describing the error
 */
inline void petsc_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw acfd::Petsc_exception(str);
}

}

#endif
