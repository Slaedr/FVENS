/**
 * @file aarray2d.cpp
 * @brief Some method implementations for the 2d arrays class.
 *
 * Part of FVENS.
 * @author Aditya Kashi
 */

#include "aarray2d.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {
namespace amat {

template <typename T>
Array2d<T>::Array2d(const Array2d<T>& other)
	: nrows{other.nrows}, ncols{other.ncols}, size{other.size},
	  elems{new T[nrows*ncols]}
{
	for(fint i = 0; i < nrows*ncols; i++)
	{
		elems[i] = other.elems[i];
	}
}

template <typename T>
Array2d<T>::Array2d(Array2d<T>&& other)
	: nrows{other.nrows}, ncols{other.ncols}, size{other.size}, elems{other.elems}
{
	other.elems = nullptr;
	other.nrows=0;
	other.ncols=0;
	other.size=0;
}

template <typename T>
Array2d<T>& Array2d<T>::operator=(const Array2d<T>& rhs)
{
#ifdef DEBUG
	if(this==&rhs) return *this;		// check for self-assignment
#endif
	nrows = rhs.nrows;
	ncols = rhs.ncols;
	size = nrows*ncols;
	delete [] elems;
	elems = new T[nrows*ncols];
	for(fint i = 0; i < nrows*ncols; i++)
	{
		elems[i] = rhs.elems[i];
	}
	return *this;
}

template <typename T>
void Array2d<T>::ones()
{
	for(fint i = 0; i < size; i++)
		elems[i] = 1;
}

template <typename T>
void Array2d<T>::setdata(const T* A, fint sz)
{
#ifdef DEBUG
	if(sz != size)
	{
		std::cout << "\nError in setdata: argument size does not match matrix size";
		return;
	}
#endif
	for(fint i = 0; i < nrows; i++)
		for(fint j = 0; j < ncols; j++)
			elems[i*ncols+j] = A[i*ncols+j];
}

template <typename T>
void Array2d<T>::mprint() const
{
	std::cout << "\n";
	for(fint i = 0; i < nrows; i++)
	{
		for(fint j = 0; j < ncols; j++)
			std::cout << std::setw(WIDTH) << std::setprecision(WIDTH/2+1) << elems[i*ncols+j];
		std::cout << std::endl;
	}
}

template <typename T>
void Array2d<T>::fprint(std::ofstream& outfile) const
{
	//outfile << '\n';
	outfile << std::setprecision(MATRIX_DOUBLE_PRECISION);
	for(fint i = 0; i < nrows; i++)
	{
		for(fint j = 0; j < ncols; j++)
			outfile << " " << elems[i*ncols+j];
		outfile << '\n';
	}
}

template <typename T>
void Array2d<T>::fread(std::ifstream& infile)
{
	infile >> nrows; infile >> ncols;
	size = nrows*ncols;
	delete [] elems;
	elems = new T[nrows*ncols];
	for(fint i = 0; i < nrows; i++)
		for(fint j = 0; j < ncols; j++)
			infile >> elems[i*ncols + j];
}

template class Array2d<freal>;
template class Array2d<fint>;

#ifdef USE_ADOLC
template class Array2d<adouble>;
#endif

template <typename T>
bool areEqual_array2d(const Array2d<T>& a, const Array2d<T>& b)
{
	if(a.rows() != b.rows()) return false;
	if(a.cols() != b.cols()) return false;
	if(a.msize() != b.msize()) return false;
	for(fint i = 0; i < a.rows(); i++)
		for(fint j = 0; j < a.cols(); j++)
			if(std::abs(a(i,j)-b(i,j)) > std::numeric_limits<T>::epsilon())
			   return false;
	return true;
}

}
}
