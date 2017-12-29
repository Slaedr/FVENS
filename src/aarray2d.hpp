/**
 * @file aarray2d.hpp
 * @brief Defines a class to manipulate 2d arrays.
 * 
 * Part of FVENS.
 * @author Aditya Kashi
 * @date Feb 10, 2015
 *
 * 2016-04-17: Removed variable storage-order. Everything is row-major now.
 * Further, all indexing is now by a_int rather than int.
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

namespace amat {
	
/// Real type
using acfd::a_real;

/// Integer type
using acfd::a_int;

const int WIDTH = 10;		// width of field for printing matrices

inline a_real dabs(a_real x)
{
	if(x < 0) return (-1.0)*x;
	else return x;
}
inline a_real minmod(a_real a, a_real b)
{
	if(a*b>0 && dabs(a) <= dabs(b)) return a;
	else if (a*b>0 && dabs(b) < dabs(a)) return b;
	else return 0.0;
}

/**
 * \class Array2d
 * \brief Stores a dense row-major matrix.
 * 
 * Notes:
 * If A is a column-major matrix, A[i][j] == A[j * nrows + i] 
 * where i is the row-index and j is the column index.
 */
template <class T>
class Array2d
{
private:
	a_int nrows;
	a_int ncols;
	a_int size;
	T* elems;

public:
	/// No-arg constructor. Note: no memory allocation!
	Array2d();

	// Full-arg constructor
	Array2d(a_int nr, a_int nc);

	Array2d(const Array2d<T>& other);

	~Array2d();

	Array2d<T>& operator=(const Array2d<T>& rhs);

	/// Separate setup function in case no-arg constructor has to be used
	/** \deprecated Please use resize() instead.
	 */
	void setup(const a_int nr, const a_int nc);
	
	/// Sets a new size for the array, deletes the contents and allocates new memory
	void resize(const a_int nr, const a_int nc);

	/// Setup without deleting earlier allocation: use in case of Array2d<t>* (pointer to Array2d<t>)
	void setupraw(a_int nr, a_int nc);
	
	/// Fill the matrix with zeros.
	void zeros();

	void ones();

	void identity();

	/// function to set matrix elements from a ROW-MAJOR array
	void setdata(const T* A, a_int sz);

	T get(const a_int i, const a_int j=0) const
	{
		assert(i < nrows);
		assert(j < ncols);
		assert(i>=0 && j>=0);
		return elems[i*ncols + j];
	}

	void set(a_int i, a_int j, T data)
	{
		assert(i < nrows);
		assert(j < ncols);
		assert(i>=0 && j>=0);
		elems[i*ncols + j] = data;
	}

	a_int rows() const { return nrows; }
	a_int cols() const { return ncols; }
	a_int msize() const { return size; }

	/// Prints the matrix to standard output.
	void mprint() const;

	/// Prints the matrix to file
	void fprint(std::ofstream& outfile) const;

	/// Reads matrix from file
	void fread(std::ifstream& infile);

	/// Getter/setter function for expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	T& operator()(const a_int x, const a_int y=0)
	{
		assert(x < nrows);
		assert(y < ncols);
		assert(x >= 0 && y >= 0);
		return elems[x*ncols + y];
	}
	
	/// Const Getter/setter function for expressions like x = A(1,2) to get the element at 1st row and 2nd column
	const T& operator()(const a_int x, const a_int y=0) const
	{
		assert(x < nrows);
		assert(x >= 0 && y >= 0);
		return elems[x*ncols + y];
	}

	/// Returns a pointer-to-const to the beginning of a row
	const T* const_row_pointer(const a_int r) const
	{
		assert(r < nrows);
		return &elems[r*ncols];
	}
	
	/// Returns a pointer to the beginning of a row
	T* row_pointer(const a_int r)
	{
		assert(r < nrows);
		return &elems[r*ncols];
	}

	T maxincol(a_int j) const
	{
		T max = get(0,j);
		for(a_int i = 0; i < nrows; i++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T maxinrow(a_int i) const
	{
		T max = get(i,0);
		for(a_int j = 0; j < nrows; j++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T max() const
	{
		T max = elems[0];
		for(a_int i = 0; i < size; i++)
			if(elems[i] > max) max = elems[i];
		return max;
	}

	T absmax() const
	{
		T max = abs(elems[0]);
		for(a_int i = 0; i < size; i++)
			if(abs(elems[i]) > max) max = abs(elems[i]);
		return max;
	}

	/// Returns the magnitude of the element with largest magnitude
	a_real dabsmax() const
	{
		a_real max = dabs((a_real)elems[0]);
		for(a_int i = 0; i < size; i++)
			if(dabs(elems[i]) > max) max = dabs(elems[i]);
		return max;
	}

	T minincol(const a_int j) const
	{
		T min = get(0,j);
		for(a_int i = 0; i < nrows; i++)
			if(min > get(i,j)) min = get(i,j);
		return min;
	}

	T mininrow(const a_int i) const
	{
		T max = get(i,0);
		for(a_int j = 0; j < nrows; j++)
			if(max > get(i,j)) max = get(i,j);
		return max;
	}

	T min() const
	{
		T min = elems[0];
		for(a_int i = 0; i < size; i++)
			if(elems[i] < min) min = elems[i];
		return min;
	}

	T average() const
	{
		T avg = 0;
		for(a_int i = 0; i < size; i++)
			avg += elems[i];
		avg = avg/size;
		return avg;
	}

	// Returns the vector 2-norm
	T l2norm() const		
	{
		T tot = 0;
		for(a_int i = 0; i < size; i++)
		{
			tot += elems[i]*elems[i];
		}
		tot = std::sqrt(tot);
		return tot;
	}

	/*
	/// Returns sum of products of respective elements of flattened arrays 
	/// containing matrix elements of this and A
	T dot_product(const Array2d<T>& A)
	{
		const T* elemsA = A.elems;
		T ans = 0;
		for(a_int i = 0; i < size; i++)
		{
			ans += elems[i]*elemsA[i];
		}
		return ans;
	}

	/// Computes 1-norm (max column-sum norm) of the matrix
	T matrixNorm_1() const
	{
		T max = 0, sum;
		a_int i,j;
		for(j = 0; j < ncols; j++)
		{
			sum = 0;
			for(i = 0; i < nrows; i++)
			{
				sum += (T)( fabs(get(i,j)) );
			}
			if(max < sum) max = sum;
		}
		return max;
	}*/
};


} //end namespace amat

#endif
