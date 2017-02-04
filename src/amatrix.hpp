/**
 * @file amatrix.hpp
 * @brief Defines a class to manipulate matrices.
 * 
 * Part of FVENS.
 * @author Aditya Kashi
 * @date Feb 10, 2015
 *
 * 2016-04-17: Removed all storage-order stuff. Everything is row-major now.
 * Further, all indexing is now by acfd_int rather than int.
 * Renamed file to amatrix.hpp from amatrix2.hpp.
 */

/**
 * \namespace amat
 * \brief Includes all array and matrix storage classes, as well as linear algebra.
 */

#ifndef __AMATRIX_H

#ifndef _GLIBCXX_IOSTREAM
#include <iostream>
#endif

#ifndef _GLIBCXX_IOMANIP
#include <iomanip>
#endif

#ifndef _GLIBCXX_FSTREAM
#include <fstream>
#endif

#ifndef _GLIBCXX_CMATH
#include <cmath>
#endif

#ifdef _OPENMP
	#ifndef OMP_H
		#include <omp.h>
		#define nthreads_m 4
	#endif
#endif

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#define __AMATRIX_H

#ifndef MATRIX_DOUBLE_PRECISION
#define MATRIX_DOUBLE_PRECISION 14
#endif

namespace amat {
	
/// Real type
using acfd::acfd_real;

/// Integer type
using acfd::acfd_int;

const int WIDTH = 10;		// width of field for printing matrices

template <class T>
class Matrix;

inline acfd_real dabs(acfd_real x)
{
	if(x < 0) return (-1.0)*x;
	else return x;
}
inline acfd_real minmod(acfd_real a, acfd_real b)
{
	if(a*b>0 && dabs(a) <= dabs(b)) return a;
	else if (a*b>0 && dabs(b) < dabs(a)) return b;
	else return 0.0;
}

template <typename T>
T determinant(const Matrix<T>& mat);

/**
 * \class Matrix
 * \brief Stores a dense row-major matrix.
 * 
 * Notes:
 * If A is a column-major matrix, A[i][j] == A[j * nrows + i] where i is the row-index and j is the column index.
 */
template <class T>
class Matrix
{
private:
	acfd_int nrows;
	acfd_int ncols;
	acfd_int size;
	T* elems;
	bool isalloc;

public:
	///No-arg constructor. Note: no memory allocation! Make sure Matrix::setup(acfd_int,acfd_int,MStype) is used.
	Matrix()
	{
		nrows = 0; ncols = 0; size = 0;
		isalloc = false;
	}

	// Full-arg constructor
	Matrix(acfd_int nr, acfd_int nc)
	{
		if(nc==0)
		{
			std::cout << "\nError: Number of columns is zero. Setting it to 1.";
			nc=1;
		}
		if(nr==0)
		{
			std::cout << "\nError: Number of rows is zero. Setting it to 1.";
			nr=1;
		}
		nrows = nr; ncols = nc;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
	}

	Matrix(const Matrix<T>& other)
	{
		nrows = other.nrows;
		ncols = other.ncols;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(acfd_int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = other.elems[i];
		}
	}

	~Matrix()
	{
		if(isalloc == true)	
			delete [] elems;
		isalloc = false;
	}

	Matrix<T>& operator=(const Matrix<T>& rhs)
	{
#ifdef DEBUG
		if(this==&rhs) return *this;		// check for self-assignment
#endif
		nrows = rhs.nrows;
		ncols = rhs.ncols;
		size = nrows*ncols;
		if(isalloc == true)
			delete [] elems;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(acfd_int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = rhs.elems[i];
		}
		return *this;
	}

	/// Separate setup function in case no-arg constructor has to be used
	void setup(acfd_int nr, acfd_int nc)
	{
		if(nc==0)
		{
			std::cout << "Matrix: setup(): Error: Number of columns is zero. Setting it to 1.\n";
			nc=1;
		}
		if(nr==0)
		{
			std::cout << "Matrix(): setup(): Error: Number of rows is zero. Setting it to 1.\n";
			nr=1;
		}
		nrows = nr; ncols = nc;
		size = nrows*ncols;
		if(isalloc == true)
			delete [] elems;
		elems = new T[nrows*ncols];
		isalloc = true;
	}

	/// Setup without deleting earlier allocation: use in case of Matrix<t>* (pointer to Matrix<t>)
	void setupraw(acfd_int nr, acfd_int nc)
	{
		//std::cout << "\nEntered setupraw";
		if(nc==0)
		{
			std::cout << "\nError: Number of columns is zero. Setting it to 1.";
			nc=1;
		}
		if(nr==0)
		{
			std::cout << "\nError: Number of rows is zero. Setting it to 1.";
			nr=1;
		}
		nrows = nr; ncols = nc;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
	}
	
	/// Fill the matrix with zeros.
	void zeros()
	{
		for(acfd_int i = 0; i < size; i++)
			elems[i] = (T)(0.0);
	}

	void ones()
	{
		for(acfd_int i = 0; i < size; i++)
			elems[i] = 1;
	}

	void identity()
	{
		T one = (T)(1);
		T zero = (T)(0);
		for(acfd_int i = 0; i < nrows; i++)
			for(acfd_int j = 0; j < ncols; j++)
				if(i==j) operator()(i,j) = one;
				else operator()(i,j) = zero;
	}

	/// function to set matrix elements from a ROW-MAJOR array
	void setdata(const T* A, acfd_int sz)
	{
#ifdef DEBUG
		if(sz != size)
		{
			std::cout << "\nError in setdata: argument size does not match matrix size";
			return;
		}
#endif
		for(acfd_int i = 0; i < nrows; i++)
			for(acfd_int j = 0; j < ncols; j++)
				elems[i*ncols+j] = A[i*ncols+j];
	}

	T get(const acfd_int i, const acfd_int j=0) const
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Matrix: get(): Index beyond array size(s)\n"; return 0; }
		if(i < 0 || j < 0) { std::cout << "Matrix: get(): Index less than 0!\n"; return 0; }
#endif
		return elems[i*ncols + j];
	}

	void set(acfd_int i, acfd_int j, T data)
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Matrix: set(): Index beyond array size(s)\n"; return; }
		if(i < 0 || j < 0) {std::cout << "Matrix: set(): Negative index!\n"; return; }
#endif
		elems[i*ncols + j] = data;
	}

	acfd_int rows() const { return nrows; }
	acfd_int cols() const { return ncols; }
	acfd_int msize() const { return size; }

	/// Prints the matrix to standard output.
	void mprint() const
	{
		std::cout << "\n";
		for(acfd_int i = 0; i < nrows; i++)
		{
			for(acfd_int j = 0; j < ncols; j++)
				std::cout << std::setw(WIDTH) << std::setprecision(WIDTH/2+1) << elems[i*ncols+j];
			std::cout << std::endl;
		}
	}

	/// Prints the matrix to file
	void fprint(std::ofstream& outfile) const
	{
		//outfile << '\n';
		outfile << std::setprecision(MATRIX_DOUBLE_PRECISION);
		for(acfd_int i = 0; i < nrows; i++)
		{
			for(acfd_int j = 0; j < ncols; j++)
				outfile << " " << elems[i*ncols+j];
			outfile << '\n';
		}
	}

	/// Reads matrix from file
	void fread(std::ifstream& infile)
	{
		infile >> nrows; infile >> ncols;
		size = nrows*ncols;
		delete [] elems;
		elems = new T[nrows*ncols];
		for(acfd_int i = 0; i < nrows; i++)
			for(acfd_int j = 0; j < ncols; j++)
				infile >> elems[i*ncols + j];
	}

	/// Getter/setter function for expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	T& operator()(const acfd_int x, const acfd_int y=0)
	{
#ifdef DEBUG
		if(x>=nrows || y>=ncols) { std::cout << "! Matrix (): Index beyond array size(s)\n"; return elems[0]; }
		if(x<0 || y<0) { std::cout << "! Matrix (): Index negative!\n"; return elems[0]; }
#endif
		return elems[x*ncols + y];
	}
	
	/// Const Getter/setter function for expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	const T& operator()(const acfd_int x, const acfd_int y=0) const
	{
#ifdef DEBUG
		if(x>=nrows || y>=ncols) { std::cout << "Matrix (): Index beyond array size(s)\n"; return elems[0]; }
		if(x<0 || y<0) { std::cout << "! Matrix (): Index negative!\n"; return elems[0]; }
#endif
		return elems[x*ncols + y];
	}

	/// Returns a pointer-to-const to the beginning of a row
	const T* row_pointer(const acfd_int r) const
	{
#ifdef DEBUG
		if(t >= nrows) { std::cout << "! Matrix: row_pointer(): Row index beyond array size!\n"; return nullptr;
#endif
		return &elems[r*ncols];
	}

	T maxincol(acfd_int j) const
	{
		T max = get(0,j);
		for(acfd_int i = 0; i < nrows; i++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T maxinrow(acfd_int i) const
	{
		T max = get(i,0);
		for(acfd_int j = 0; j < nrows; j++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T max() const
	{
		T max = elems[0];
		for(acfd_int i = 0; i < size; i++)
			if(elems[i] > max) max = elems[i];
		return max;
	}

	T absmax() const
	{
		T max = abs(elems[0]);
		for(acfd_int i = 0; i < size; i++)
			if(abs(elems[i]) > max) max = abs(elems[i]);
		return max;
	}

	/// Returns the magnitude of the element with largest magnitude
	acfd_real dabsmax() const
	{
		acfd_real max = dabs((acfd_real)elems[0]);
		for(acfd_int i = 0; i < size; i++)
			if(dabs(elems[i]) > max) max = dabs(elems[i]);
		return max;
	}

	T minincol(acfd_int j) const
	{
		T min = get(0,j);
		for(acfd_int i = 0; i < nrows; i++)
			if(min > get(i,j)) min = get(i,j);
		return min;
	}

	T mininrow(acfd_int i) const
	{
		T max = get(i,0);
		for(acfd_int j = 0; j < nrows; j++)
			if(max > get(i,j)) max = get(i,j);
		return max;
	}

	T min() const
	{
		T max = elems[0];
		for(acfd_int i = 0; i < size; i++)
			if(elems[i] < max) max = elems[i];
		return max;
	}

	T average() const
	{
		T avg = 0;
		for(acfd_int i = 0; i < size; i++)
			avg += elems[i];
		avg = avg/size;
		return avg;
	}

	// sums the square of all elements in the matrix and returns the square root of this sum
	T l2norm() const		
	{
		T tot = 0;
		for(acfd_int i = 0; i < size; i++)
		{
			tot += elems[i]*elems[i];
		}
		tot = std::sqrt(tot);
		return tot;
	}

	/// function to return a sub-matrix of this matrix
	Matrix<T> sub(acfd_int startr, acfd_int startc, acfd_int offr, acfd_int offc) const
	{
		Matrix<T> B(offr, offc);
		for(acfd_int i = 0; i < offr; i++)
			for(acfd_int j = 0; j < offc; j++)
				B(i,j) = elems[(startr+i)*ncols + startc + j];
		return B;
	}

	/// Function that returns a given column of the matrix as a row-major matrix
	Matrix<T> col(acfd_int j) const
	{
		Matrix<T> b(nrows, 1);
		for(acfd_int i = 0; i < nrows; i++)
			b(i,0) = elems[i*ncols + j];
		return b;
	}

	Matrix<T> row(acfd_int i) const
	{
		Matrix<T> b(1, ncols);
		for(acfd_int j = 0; j < ncols; j++)
			b(0,j) = elems[i*ncols + j];
		return b;
	}

	/*//Function to return a reference to a given column of the matrix
	Matrix<T>& colr(acfd_int j)
	{
		//Matrix<T>* b(nrows, 1);
		Matrix<T>* b; b->elems.reserve(nrows);
		for(acfd_int i = 0; i < nrows; i++)
			b.elems[i] = &elems[i*ncols + j];
		return *b;
	} */

	/// Function for replacing a column of the matrix with a vector. NOTE: No check for whether b is really a vector - which it must be.
	void replacecol(acfd_int j, Matrix<T> b)
	{
#ifdef DEBUG
		if(b.cols() != 1 || b.rows() != nrows) { std::cout << "\nSize error in replacecol"; return; }
#endif
		for(acfd_int i = 0; i < nrows; i++)
			elems[i*ncols + j] = b.elems[i];
	}

	/// Function for replacing a row
	void replacerow(acfd_int i, Matrix<T> b)
	{
#ifdef DEBUG
		if(b.cols() != ncols || b.rows() != 1) { std::cout << "\nSize error in replacerow"; return; }
#endif
		for(acfd_int j = 0; j < ncols; j++)
			elems[i*ncols + j] = b.elems[j];
	}

	/// Returns the transpose
	Matrix<T> trans() const
	{
		Matrix<T> t(ncols, nrows);
		for(acfd_int i = 0; i < ncols; i++)
			for(acfd_int j = 0; j < nrows; j++)
				t(i,j) = get(j,i);
		return t;
	}

	/// Multiply a matrix by a scalar. Note: only expressions of type A*3 work, not 3*A
	Matrix<T> operator*(T num)
	{
		Matrix<T> A(nrows,ncols);
		acfd_int i;

		for(i = 0; i < A.size; i++)
			A.elems[i] = elems[i] * num;
		return A;
	}

	/**	The matrix addition and subtraction operators are inefficient! Do not use in long loops. */
	Matrix<T> operator+(Matrix<T> B) const
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Matrix: Addition cannot be performed due to incompatible sizes\n";
			Matrix<T> C(1,1);
			return C;
		}
#endif
		Matrix<T> C(nrows, ncols);
		acfd_int i;

		for(i = 0; i < C.size; i++)
			C.elems[i] = elems[i] + B.elems[i];
		return C;
	}

	Matrix<T> operator-(Matrix<T> B) const
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Matrix: Subtraction cannot be performed due to incompatible sizes\n";
			Matrix<T> C(1,1);
			return C;
		}
#endif
		Matrix<T> C(nrows, ncols);

		for(acfd_int i = 0; i < C.size; i++)
			C.elems[i] = elems[i] - B.elems[i];
		return C;
	}

	Matrix<T> operator*(Matrix<T> B)
	{
		Matrix<acfd_real> C(nrows, B.cols());
		C.zeros();
#ifdef DEBUG
		if(ncols != B.rows())
		{
			std::cout << "! Matrix: Multiplication cannot be performed - incompatible sizes!\n";
			return C;
		}
#endif
		for(acfd_int i = 0; i < nrows; i++)
			for(acfd_int j = 0; j < B.cols(); j++)
				for(acfd_int k = 0; k < ncols; k++)
					C(i,j) += get(i,k) * B.get(k,j);
					//C.set( C.get(i,j) + get(i,k)*B.get(k,j), i,j );

		return C;
	}

	/// Returns sum of products of respective elements of flattened arrays containing matrix elements of this and A
	T dot_product(const Matrix<T>& A)
	{
		T* elemsA = A.elems;
		#ifdef _OPENMP
		T* elems = this->elems;
		acfd_int size = this->size;
		#endif
		acfd_int i;
		T ans = 0;
		//#pragma omp parallel for if(size >= 64) default(none) private(i) shared(elems,elemsA,size) reduction(+: ans) num_threads(nthreads_m)
		for(i = 0; i < size; i++)
		{
			T temp = elems[i]*elemsA[i];
			ans += temp;
		}
		return ans;
	}

	/// Computes 1-norm (max column-sum norm) of the matrix
	T matrixNorm_1() const
	{
		T max = 0, sum;
		acfd_int i,j;
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
	}

	friend T determinant<>(const Matrix<T>& mat);
};


} //end namespace amat

#endif
