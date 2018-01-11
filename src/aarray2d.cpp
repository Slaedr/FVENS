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

namespace amat {

/// Separate setup function in case no-arg constructor has to be used
/** \deprecated Please use resize() instead.
 */
template <typename T>
void Array2d<T>::setup(const a_int nr, const a_int nc)
{
	if(nc==0)
	{
		std::cout << "Array2d: setup(): ! Error: Number of columns is zero!\n";
		return;
	}
	if(nr==0)
	{
		std::cout << "Array2d(): setup(): ! Error: Number of rows is zero!\n";
		return;
	}
	nrows = nr; ncols = nc;
	size = nrows*ncols;
	delete [] elems;
	elems = new T[nrows*ncols];
}

/// Setup without deleting earlier allocation: use in case of Array2d<t>* (pointer to Array2d<t>)
template <typename T>
void Array2d<T>::setupraw(const a_int nr, const a_int nc)
{
	assert(nc>0);
	assert(nr>0);
		
	nrows = nr; ncols = nc;
	size = nrows*ncols;
	delete [] elems;
	elems = new T[nrows*ncols];
}

template <typename T>
void Array2d<T>::ones()
{
	for(a_int i = 0; i < size; i++)
		elems[i] = 1;
}

template <typename T>
void Array2d<T>::setdata(const T* A, a_int sz)
{
#ifdef DEBUG
	if(sz != size)
	{
		std::cout << "\nError in setdata: argument size does not match matrix size";
		return;
	}
#endif
	for(a_int i = 0; i < nrows; i++)
		for(a_int j = 0; j < ncols; j++)
			elems[i*ncols+j] = A[i*ncols+j];
}

template <typename T>
void Array2d<T>::mprint() const
{
	std::cout << "\n";
	for(a_int i = 0; i < nrows; i++)
	{
		for(a_int j = 0; j < ncols; j++)
			std::cout << std::setw(WIDTH) << std::setprecision(WIDTH/2+1) << elems[i*ncols+j];
		std::cout << std::endl;
	}
}

template <typename T>
void Array2d<T>::fprint(std::ofstream& outfile) const
{
	//outfile << '\n';
	outfile << std::setprecision(MATRIX_DOUBLE_PRECISION);
	for(a_int i = 0; i < nrows; i++)
	{
		for(a_int j = 0; j < ncols; j++)
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
	for(a_int i = 0; i < nrows; i++)
		for(a_int j = 0; j < ncols; j++)
			infile >> elems[i*ncols + j];
}

template class Array2d<a_real>;
template class Array2d<a_int>;

}
