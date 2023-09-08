/** \file aerrorhandling.hpp
 * \brief Exceptions and other error handling routines for FVENS
 * \author Aditya Kashi
 */

#ifndef AERRORHANDLING_H
#define AERRORHANDLING_H

#include <stdexcept>
#include <string>
#include <fstream>

namespace fvens {

/// Exception to throw on errors related to numerics
class Numerical_error : public std::logic_error
{
public:
	/// Construct a numerical error with a message
	Numerical_error(const std::string& msg);
};

/// Exception thrown when some solver does not meet the required tolerance
class Tolerance_error : public Numerical_error
{
public:
	Tolerance_error(const std::string& msg);
};

/// Exception thrown due to return codes from MPI functions
class MPI_exception : public std::runtime_error
{
public:
	MPI_exception(const std::string& msg);
	MPI_exception(const char *const msg);
};
	
/// An exception to throw for errors from PETSc; takes a custom message
class Petsc_exception : public std::runtime_error
{
public:
	Petsc_exception(int ierr, const std::string& msg);
	Petsc_exception(int ierr, const char *msg);
};

/// Exception thrown when a required input was not provided
class InputNotGivenError : public std::runtime_error
{
public:
	InputNotGivenError(const std::string& msg);
};

/// Exception thrown when a user-supplied option is invalid
class UnsupportedOptionError : public std::runtime_error
{
public:
	UnsupportedOptionError(const std::string& msg);
};

/// Throw an error from an error code
/** \param ierr An expression which, if true, triggers the exception
 * \param str A short string message describing the error
 */
inline void fvens_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw std::runtime_error(str);
}

/// Throw an error from an error code related to MPI
/** \param ierr an expression which, if true, triggers the exception
 * \param str a short string message describing the error
 */
inline void mpi_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw fvens::MPI_exception(str);
}

/// throw an error from an error code related to petsc
/** \param ierr an expression which, if true, triggers the exception
 * \param str a short string message describing the error
 */
inline void petsc_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw fvens::Petsc_exception(ierr, str);
}

/// Opens a file for reading but aborts in case of an error
void open_file_toRead(const std::string file, std::ifstream& fin);

/// Opens a file for writing but throws in case of an error
void open_file_toWrite(const std::string file, std::ofstream& fout);

}

#endif
