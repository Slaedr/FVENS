/**
 * @file aarray2d.hpp
 * @brief Defines a class to manipulate 2d arrays.
 * 
 * Part of FVENS.
 * @author Aditya Kashi
 * @date Feb 10, 2015
 */

/**
 * \namespace amat
 * \brief Includes matrix and some linear algebra classes.
 */

#ifndef AARRAY2D_H
#define AARRAY2D_H

#include <cassert>
#include "aconstants.hpp"

#ifndef MATRIX_DOUBLE_PRECISION
#define MATRIX_DOUBLE_PRECISION 14
#endif

namespace fvens {
namespace amat {
	
/// Real type
using fvens::freal;

/// Integer type
using fvens::fint;

const int WIDTH = 10;		// width of field for printing matrices

/**
 * \class Array2d
 * \brief Stores a dense 2D row-major array.
 * \deprecated Use std::vector instead, if possible.
 */
template <class T>
class Array2d
{
protected:
	fint nrows;           ///< Number of rows
	fint ncols;           ///< Number of columns
	fint size;            ///< Total number of entries
	T* elems;              ///< Raw array of entries

public:
	/// No-arg constructor. Note: no memory allocation!
	Array2d() : nrows{0}, ncols{0}, size{0}, elems{nullptr}
	{ }

	/// Allocate some storage
	Array2d(const fint nr, const fint nc)
	{
		assert(nc>=0);
		assert(nr>=0);
		
		nrows = nr; ncols = nc;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
	}

	/// Deep copy
	Array2d(const Array2d<T>& other);

	/// Move constructor
	/** Performs a shallow copy and then nulls the other array to a 0x0 array.
	 */
	Array2d(Array2d<T>&& other);

	~Array2d()
	{
		delete [] elems;
	}

	/// Deep copy
	Array2d<T>& operator=(const Array2d<T>& rhs);
	
	/// Sets a new size for the array, deletes the contents and allocates new memory
	void resize(const fint nr, const fint nc)
	{
		assert(nc>=0);
		assert(nr>=0);
		
		nrows = nr; ncols = nc;
		size = nrows*ncols;
		delete [] elems;
		elems = new T[nrows*ncols];
	}
	
	/// Fill the array with zeros.
	void zeros()
	{
		for(fint i = 0; i < size; i++)
			elems[i] = (T)(0);
	}

	/// Fill the array with ones
	void ones();

	/// function to copy matrix elements from a ROW-MAJOR array
	void setdata(const T* A, fint sz);

	/// Getter function \sa operator()
	T get(const fint i, const fint j=0) const
	{
		assert(i < nrows);
		assert(j < ncols);
		assert(i>=0 && j>=0);
		return elems[i*ncols + j];
	}

	fint rows() const { return nrows; }
	fint cols() const { return ncols; }
	fint msize() const { return size; }

	/// Getter/setter function for expressions like A(1,2) = 141
	///  to set the element at 1st row and 2nd column to 141
	T& operator()(const fint x, const fint y=0)
	{
		assert(x < nrows);
		assert(y < ncols);
		assert(x >= 0 && y >= 0);
		return elems[x*ncols + y];
	}
	
	/// Const getter function for expressions like x = A(1,2) to get the element at 1st row and 2nd column
	const T& operator()(const fint x, const fint y=0) const
	{
		assert(x < nrows);
		assert(y < ncols);
		assert(x >= 0 && y >= 0);
		return elems[x*ncols + y];
	}

	/// Returns a pointer-to-const to the beginning of a row
	const T* const_row_pointer(const fint r) const
	{
		assert(r < nrows);
		return &elems[r*ncols];
	}
	
	/// Returns a pointer to the beginning of a row
	T* row_pointer(const fint r)
	{
		assert(r < nrows);
		return &elems[r*ncols];
	}
	
	/// Prints the matrix to standard output.
	void mprint() const;

	/// Prints the matrix to file
	void fprint(std::ofstream& outfile) const;

	/// Reads matrix from file
	void fread(std::ifstream& infile);
};

template <typename T>
bool areEqual_array2d(const Array2d<T>& a, const Array2d<T>& b);

/// An immutable 2D view of a raw array
template <typename T>
class Array2dView
{
protected:
	fint nrows;             ///< Number of rows
	fint ncols;             ///< Number of columns
	const T *const elems;    ///< Pointer to data
public:
	/// Wrapper constructor - does not take ownership and does not delete storage once done
	Array2dView(const T *const array, const fint nr, const fint nc)
		: nrows{nr}, ncols{nc}, elems{array}
	{ }

	/// Accessor
	const T& operator()(const fint x, const fint y=0) const
	{
		assert(x < nrows);
		assert(y < ncols);
		assert(x >= 0 && y >= 0);
		return elems[x*ncols + y];
	}

	fint rows() const { return nrows; }
	fint cols() const { return ncols; }
};

/// A mutable 2D view of a raw array
template <typename T>
class Array2dMutableView
{
protected:
	fint nrows;
	fint ncols;
	T *const elems;
public:
	/// Wrapper constructor - does not take ownership and does not delete storage once done
	Array2dMutableView(T *const array, const fint nr, const fint nc)
		: nrows{nr}, ncols{nc}, elems{array}
	{ }

	/// Accessor
	T& operator()(const fint x, const fint y=0)
	{
		assert(x < nrows);
		assert(y < ncols);
		assert(x >= 0 && y >= 0);
		return elems[x*ncols + y];
	}

	fint rows() const { return nrows; }
	fint cols() const { return ncols; }
};


} //end namespace amat
} // namespace fvens

#endif
