/*
This file defines a class Matrix to store a matrix in column-major or row-major storage.

Aditya Kashi
Feb 10, 2015

Notes:
If A is a column-major matrix, A[i][j] == A[j * nrows + i] where i is the row-index and j is the column index.

TODO: (Not really needed) Make a Vector class, maybe as a sub-class of Matrix
*/

#ifndef _GLIBCXX_IOSTREAM
#include <iostream>
#endif

#ifndef _GLIBCXX_IOMANIP
#include <iomanip>
#endif

#ifndef _GLIBCXX_ARRAY
#include <array>
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
#define nthreads_m 8
#endif
#endif

#define __AMATRIX2_H

//using namespace std;

const int WIDTH = 12;		// width of field for printing matrices

namespace amat {

double dabs(double x)
{
	if(x < 0) return (-1.0)*x;
	else return x;
}
double minmod(double a, double b)
{
	if(a*b>0 && dabs(a) <= dabs(b)) return a;
	else if (a*b>0 && dabs(b) < dabs(a)) return b;
	else return 0.0;
}

enum MStype {ROWMAJOR, COLMAJOR};

/// Basic matrix-handling class template; this definition is never used
template <class T, MStype storage> class Matrix
{
private:
	int nrows;
	int ncols;
	int size;
	T* elems;
	bool isalloc;

public:
	//No-arg constructor. Note: no memory allocation! Make sure Matrix::setup(int,int,MStype) is used.
	Matrix()
	{
		nrows = 0; ncols = 0; size = 0;
		isalloc = false;
	}

	// Full-arg constructor
	Matrix(int nr, int nc)
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

	~Matrix()
	{
		if(isalloc == true)	
			delete [] elems;
		isalloc = false;
	}
};

/// Matrix template specialized to row-major storage
template <class T> class Matrix<T, ROWMAJOR>
{
private:
	int nrows;
	int ncols;
	int size;
	T* elems;
	bool isalloc;

public:
	/// No-arg constructor. Note: no memory allocation! Make sure Matrix::setup(int,int) is used.
	Matrix()
	{
		nrows = 0; ncols = 0; size = 0;
		isalloc = false;
	}

	// Full-arg constructor
	Matrix(int nr, int nc)
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
		nrows = nr; ncols = nc; storage = st;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
	}

	/// Copies a rowmajor array into this one
	Matrix(const Matrix<T, ROWMAJOR>& other)
	{
		nrows = other.nrows;
		ncols = other.ncols;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = other.elems[i];
		}
	}
	
	/// Copies a column-major array into this one - expensive!
	Matrix(const Matrix<T, COLMAJOR>& other)
	{
		nrows = other.nrows;
		ncols = other.ncols;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				elems[i*ncols+j] = other.elems[j*nrows+i];
	}

	~Matrix()
	{
		if(isalloc == true)	
			delete [] elems;
		isalloc = false;
	}

	Matrix<T,ROWMAJOR>& operator=(Matrix<T,ROWMAJOR> rhs)
	{
#ifdef DEBUG
		if(this==&rhs) return *this;		// check for self-assignment
#endif
		nrows = rhs.nrows;
		ncols = rhs.ncols;
		storage = rhs.storage;
		size = nrows*ncols;
		if(isalloc == true)
			delete [] elems;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = rhs.elems[i];
		}
		return *this;
	}
	
	Matrix<T,ROWMAJOR>& operator=(Matrix<T,COLMAJOR> rhs)
	{
#ifdef DEBUG
		if(this==&rhs) return *this;		// check for self-assignment
#endif
		nrows = rhs.nrows;
		ncols = rhs.ncols;
		storage = rhs.storage;
		size = nrows*ncols;
		if(isalloc == true)
			delete [] elems;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				elems[i*ncols+j] = other.elems[j*nrows+i];
		return *this;
	}

	//Separate setup function in case no-arg constructor has to be used
	void setup(int nr, int nc)
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

	// no deleting earlier allocation: use in case of Matrix<t>* (pointer to Matrix<t>)
	void setupraw(int nr, int nc)
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

	void zeros()
	{
		for(int i = 0; i < size; i++)
			elems[i] = (T)(0.0);
	}

	void ones()
	{
		for(int i = 0; i < size; i++)
			elems[i] = 1;
	}

	void identity()
	{
		T one = (T)(1);
		T zero = (T)(0);
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				if(i==j) operator()(i,j) = one;
				else operator()(i,j) = zero;
	}

	// function to set matrix elements from a ROW-MAJOR array
	void setdata(const T* A, int sz)
	{
#ifdef DEBUG
		if(sz != size)
		{
			std::cout << "\nError in setdata: argument size does not match matrix size";
			return;
		}
#endif
			for(int i = 0; i < nrows; i++)
				for(int j = 0; j < ncols; j++)
					elems[i*ncols+j] = A[i*ncols+j];
	}

	T get(int i, int j=0) const
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Matrix: get(): Index beyond array size(s)\n"; return 0; }
		if(i < 0 || j < 0) {std::cout << "Matrix: get(): Negative index!\n"; return 0; }
#endif
		return elems[i*ncols + j];
	}

	void set(int i, int j, T data)
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Matrix: set(): Index beyond array size(s)\n"; return; }
		if(i < 0 || j < 0) {std::cout << "Matrix: set(): Negative index!\n"; return 0; }
#endif
		elems[i*ncols + j] = data;
	}

	int rows() const { return nrows; }
	int cols() const { return ncols; }
	int msize() const { return size; }

	void mprint() const
	{
		std::cout << "\n";
		for(int i = 0; i < nrows; i++)
		{
			for(int j = 0; j < ncols; j++)
				std::cout << std::setw(WIDTH) << std::setprecision(WIDTH/2+1) << elems[i*ncols+j];
			std::cout << std::endl;
		}
	}

	void fprint(std::ofstream& outfile) const
	{
		for(int i = 0; i < nrows; i++)
		{
			for(int j = 0; j < ncols; j++)
				outfile << " " << elems[i*ncols+j];
			outfile << '\n';
		}
	}

	void fread(std::ifstream& infile)
	{
		infile >> nrows; infile >> ncols;
		size = nrows*ncols;
		storage = ROWMAJOR;
		delete [] elems;
		elems = new T[nrows*ncols];
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				infile >> elems[i*ncols + j];
	}

	// For expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	T& operator()(int x, int y=0)
	{
#ifdef DEBUG
		if(x>=nrows || y>=ncols) { std::cout << "Matrix (): Index beyond array size(s)\n"; return elems[0]; }
#endif
		return elems[x*ncols + y];
	}

	T maxincol(int j) const
	{
		T max = get(0,j);
		for(int i = 0; i < nrows; i++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T maxinrow(int i) const
	{
		T max = get(i,0);
		for(int j = 0; j < nrows; j++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T max() const
	{
		T max = elems[0];
		for(int i = 0; i < size; i++)
			if(elems[i] > max) max = elems[i];
		return max;
	}

	T absmax() const
	{
		T max = abs(elems[0]);
		for(int i = 0; i < size; i++)
			if(abs(elems[i]) > max) max = abs(elems[i]);
		return max;
	}

	double dabsmax() const
	{
		double max = dabs((double)elems[0]);
		for(int i = 0; i < size; i++)
			if(dabs(elems[i]) > max) max = dabs(elems[i]);
		return max;
	}

	T minincol(int j) const
	{
		T min = get(0,j);
		for(int i = 0; i < nrows; i++)
			if(min > get(i,j)) min = get(i,j);
		return min;
	}

	T mininrow(int i) const
	{
		T max = get(i,0);
		for(int j = 0; j < nrows; j++)
			if(max > get(i,j)) max = get(i,j);
		return max;
	}

	T min() const
	{
		T max = elems[0];
		for(int i = 0; i < size; i++)
			if(elems[i] < max) max = elems[i];
		return max;
	}

	T average() const
	{
		T avg = 0;
		for(int i = 0; i < size; i++)
			avg += elems[i];
		avg = avg/size;
		return avg;
	}

	T l2norm() const		// sums the square of all elements in the matrix and returns the square root of this sum
	{
		T tot = 0;
		for(int i = 0; i < size; i++)
		{
			tot += elems[i]*elems[i];
		}
		tot = std::sqrt(tot);
		return tot;
	}

	// function to return a sub-matrix of this matrix
	Matrix<T,ROWMAJOR> sub(int startr, int startc, int offr, int offc) const
	{
		Matrix<T,ROWMAJOR> B(offr, offc);
		for(int i = 0; i < offr; i++)
			for(int j = 0; j < offc; j++)
				B(i,j) = elems[(startr+i)*ncols + startc + j];
		return B;
	}

	//Function that returns a given column of the matrix as a row-major matrix
	Matrix<T,ROWMAJOR> col(int j) const
	{
		Matrix<T,ROWMAJOR> b(nrows, 1);
			for(int i = 0; i < nrows; i++)
				b(i,0) = elems[i*ncols + j];
		return b;
	}

	Matrix<T,ROWMAJOR> row(int i) const
	{
		Matrix<T> b(1, ncols);
		for(int j = 0; j < ncols; j++)
			b(0,j) = elems[i*ncols + j];
		return b;
	}

	/// Function to return a pointer to a given row of the matrix (not tested!)
	T* rowr(int i)
	{
		T* ptr = &elems[i*ncols];
		return ptr;
	}

	/// Function for replacing a column of the matrix with a vector. NOTE: No check for whether b is really a vector - which it must be.
	void replacecol(int j, Matrix<T> b)
	{
#ifdef DEBUG
		if(b.cols() != 1 || b.rows() != nrows) { std::cout << "\nSize error in replacecol"; return; }
#endif
		for(int i = 0; i < nrows; i++)
			elems[i*ncols + j] = b.elems[i];
	}

	/// Function for replacing a row
	void replacerow(int i, Matrix<T> b)
	{
#ifdef DEBUG
		if(b.cols() != ncols || b.rows() != 1) { std::cout << "\nSize error in replacerow"; return; }
#endif
		for(int j = 0; j < ncols; j++)
			elems[i*ncols + j] = b.elems[j];
	}

	//transpose
	Matrix<T,ROWMAJOR> trans() const
	{
		Matrix<T,ROWMAJOR> t(ncols, nrows);
		for(int i = 0; i < ncols; i++)
			for(int j = 0; j < nrows; j++)
				t(i,j) = get(j,i);
		return t;
	}

	/// Multiply a matrix by a scalar. Note: only expressions of type A*3 work, not 3*A
	template<MStype stor> Matrix<T,stor> operator*(T num)
	{
		Matrix<T,stor> A(nrows,ncols);
		int i;

		for(i = 0; i < A.size; i++)
			A.elems[i] = elems[i] * num;
		return A;
	}

	Matrix<T,ROWMAJOR> operator+(Matrix<T,ROWMAJOR> B)
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Matrix: Addition cannot be performed due to incompatible sizes\n";
			Matrix<T,retstor> C(1,1);
			return C;
		}
#endif
		Matrix<T,ROWMAJOR> C(nrows, ncols);
		int i;

		for(i = 0; i < nrows*ncols; i++)
			C.elems[i] = elems[i] + B.elems[i];
		return C;
	}

	Matrix<T,ROWMAJOR> operator-(Matrix<T,ROWMAJOR> B)
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

		for(int i = 0; i < C.size; i++)
			C.elems[i] = elems[i] - B.elems[i];
		return C;
	}

	Matrix<T,ROWMAJOR> operator*(Matrix<T,ROWMAJOR> B)
	{
		Matrix<T,ROWMAJOR> C(nrows, B.cols());
		C.zeros();
#ifdef DEBUG
		if(ncols != B.rows())
		{
			std::cout << "! Matrix: Multiplication cannot be performed - incompatible sizes!\n";
			return C;
		}
#endif
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < B.cols(); j++)
				for(int k = 0; k < ncols; k++)
					C(i,j) += get(i,k) * B.get(k,j);
					//C.set( C.get(i,j) + get(i,k)*B.get(k,j), i,j );

		return C;
	}

	/// Returns sum of products of respective elements of flattened arrays containing matrix elements of this and A
	T dot_product(const Matrix<T,ROWMAJOR>& A)
	{
		T* elemsA = A.elems;
		#ifdef _OPENMP
		T* elems = this->elems;
		int size = this->size;
		#endif
		int i;
		double ans = 0;
		#pragma omp parallel for if(size >= 1024) default(none) private(i) shared(elems,elemsA,size) reduction(+: ans) num_threads(nthreads_m)
		for(i = 0; i < size; i++)
		{
			T temp = elems[i]*elemsA[i];
			//#pragma omp critical (omp_dot)
			ans += temp;
		}
		return ans;
	}
};

template<class T>
class Matrix<T, COLMAJOR>
{
private:
	int nrows;
	int ncols;
	int size;
	T* elems;
	bool isalloc;

public:
	//No-arg constructor. Note: no memory allocation! Make sure Matrix::setup(int,int,MStype) is used.
	Matrix()
	{
		nrows = 0; ncols = 0; size = 0;
		isalloc = false;
	}

	// Full-arg constructor
	Matrix(int nr, int nc)
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

	Matrix(const Matrix<T,COLMAJOR>& other)
	{
		nrows = other.nrows;
		ncols = other.ncols;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = other.elems[i];
		}
	}
	
	Matrix(const Matrix<T,ROWMAJOR>& other)
	{
		nrows = other.nrows;
		ncols = other.ncols;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				elems[j*nrows+i] = other.elems[i*ncols+j];
	}

	~Matrix()
	{
		if(isalloc == true)	
			delete [] elems;
		isalloc = false;
	}

	Matrix<T,COLMAJOR>& operator=(Matrix<T,COLMAJOR> rhs)
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
		for(int i = 0; i < nrows*ncols; i++)
			elems[i] = rhs.elems[i];
		return *this;
	}
	
	Matrix<T,COLMAJOR>& operator=(Matrix<T,ROWMAJOR> rhs)
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
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				elems[j*nrows+i] = other.elems[i*ncols+j];
		return *this;
	}

	//Separate setup function in case no-arg constructor has to be used
	void setup(int nr, int nc, MStype st=ROWMAJOR)
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

	// no deleting earlier allocation: use in case of Matrix<t>* (pointer to Matrix<t>)
	void setupraw(int nr, int nc, MStype st)
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

	void zeros()
	{
		for(int i = 0; i < size; i++)
			elems[i] = (T)(0.0);
	}

	void ones()
	{
		for(int i = 0; i < size; i++)
			elems[i] = 1;
	}

	void identity()
	{
		T one = (T)(1);
		T zero = (T)(0);
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				if(i==j) operator()(i,j) = one;
				else operator()(i,j) = zero;
	}

	// function to set matrix elements from a ROW-MAJOR std::vector
	void setdata(const T* A, int sz)
	{
#ifdef DEBUG
		if(sz != size)
		{
			std::cout << "\nError in setdata: argument size does not match matrix size";
			return;
		}
#endif
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				elems[j*nrows+i] = A[i*ncols+j];
	}

	T get(int i, int j=0) const
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Matrix: get(): Index beyond array size(s)\n"; return 0; }
		if(i < 0 || j < 0) {std::cout << "Matrix: get(): Negative index!\n"; return 0; }
#endif
		return elems[j*nrows + i];
	}

	void set(int i, int j, T data)
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Matrix: set(): Index beyond array size(s)\n"; return; }
		if(i < 0 || j < 0) {std::cout << "Matrix: set(): Negative index!\n"; return 0; }
#endif
		elems[j*nrows + i] = data;
	}

	int rows() const { return nrows; }
	int cols() const { return ncols; }
	int msize() const { return size; }

	void mprint() const
	{
		std::cout << "\n";
		for(int i = 0; i < nrows; i++)
		{
			for(int j = 0; j < ncols; j++)
				std::cout << std::setw(WIDTH) << elems[j*nrows+i];
			std::cout << std::endl;
		}
	}

	void fprint(std::ofstream& outfile) const
	{
		for(int i = 0; i < nrows; i++)
		{
			for(int j = 0; j < ncols; j++)
				outfile << " " << elems[j*nrows+i];
			outfile << '\n';
		}
	}

	void fread(std::ifstream& infile)
	{
		infile >> nrows; infile >> ncols;
		size = nrows*ncols;
		delete [] elems;
		elems = new T[nrows*ncols];
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
				infile >> elems[j*nrows + i];
	}

	// For expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	T& operator()(int x, int y=0)
	{
#ifdef DEBUG
		if(x>=nrows || y>=ncols) { std::cout << "Matrix (): Index beyond array size(s)\n"; return elems[0]; }
#endif
		return elems[y*nrows + x];
	}

	T maxincol(int j) const
	{
		T max = get(0,j);
		for(int i = 0; i < nrows; i++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T maxinrow(int i) const
	{
		T max = get(i,0);
		for(int j = 0; j < nrows; j++)
			if(max < get(i,j)) max = get(i,j);
		return max;
	}

	T max() const
	{
		T max = elems[0];
		for(int i = 0; i < size; i++)
			if(elems[i] > max) max = elems[i];
		return max;
	}

	T absmax() const
	{
		T max = abs(elems[0]);
		for(int i = 0; i < size; i++)
			if(abs(elems[i]) > max) max = abs(elems[i]);
		return max;
	}

	double dabsmax() const
	{
		double max = dabs((double)elems[0]);
		for(int i = 0; i < size; i++)
			if(dabs(elems[i]) > max) max = dabs(elems[i]);
		return max;
	}

	T minincol(int j) const
	{
		T min = get(0,j);
		for(int i = 0; i < nrows; i++)
			if(min > get(i,j)) min = get(i,j);
		return min;
	}

	T mininrow(int i) const
	{
		T max = get(i,0);
		for(int j = 0; j < nrows; j++)
			if(max > get(i,j)) max = get(i,j);
		return max;
	}

	T min() const
	{
		T max = elems[0];
		for(int i = 0; i < size; i++)
			if(elems[i] < max) max = elems[i];
		return max;
	}

	T average() const
	{
		T avg = 0;
		for(int i = 0; i < size; i++)
			avg += elems[i];
		avg = avg/size;
		return avg;
	}

	T l2norm() const		// sums the square of all elements in the matrix and returns the square root of this sum
	{
		T tot = 0;
		for(int i = 0; i < size; i++)
		{
			tot += elems[i]*elems[i];
		}
		tot = std::sqrt(tot);
		return tot;
	}

	// function to return a sub-matrix of this matrix
	Matrix<T,COLMAJOR> sub(int startr, int startc, int offr, int offc) const
	{
		Matrix<T,COLMAJOR> B(offr, offc);
		for(int i = 0; i < offr; i++)
			for(int j = 0; j < offc; j++)
				B(i,j) = elems[(startc+j)*nrows + startr + i];
		return B;
	}

	//Function that returns a given column of the matrix as a column-major matrix
	Matrix<T,COLMAJOR> col(int j) const
	{
		Matrix<T,COLMAJOR> b(nrows, 1);
		for(int i = 0; i < nrows; i++)
			b(i,0) = elems[j*nrows + i];
		return b;
	}

	/// Gives a row as a column-major matrix...
	Matrix<T,COLMAJOR> row(int i) const
	{
		Matrix<T,COLMAJOR> b(1, ncols);
		for(int j = 0; j < ncols; j++)
			b(0,j) = elems[j*nrows + i];
		return b;
	}

	/// Function to return a reference to a given column of the matrix
	T* colr(int j)
	{
		T* ptr = &elems[j*nrows];
		return ptr;
	}

	// Function for replacing a column of the matrix with a vector. NOTE: No check for whether b is really a vector - which it must be.
	void replacecol(int j, Matrix<T> b)
	{
#ifdef DEBUG
		if(b.cols() != 1 || b.rows() != nrows) { std::cout << "\nSize error in replacecol"; return; }
#endif
		for(int i = 0; i < nrows; i++)
			elems[j*nrows + i] = b.elems[i];
	}

	//Function for replacing a row
	void replacerow(int i, Matrix<T> b)
	{
#ifdef DEBUG
		if(b.cols() != ncols || b.rows() != 1) { std::cout << "\nSize error in replacerow"; return; }
#endif
		for(int j = 0; j < ncols; j++)
			elems[j*nrows + i] = b.elems[j];
	}

	//transpose
	Matrix<T,COLMAJOR> trans() const
	{
		Matrix<T,COLMAJOR> t(ncols, nrows);
		for(int i = 0; i < ncols; i++)
			for(int j = 0; j < nrows; j++)
				t(i,j) = get(j,i);
		return t;
	}

	// Multiply a matrix by a scalar. Note: only expressions of type A*3 work, not 3*A
	template <MStype storage> Matrix<T,storage> operator*(T num)
	{
		Matrix<T,storage> A(nrows,ncols);
		int i;

		for(i = 0; i < A.size; i++)
			A.elems[i] = elems[i] * num;
		return A;
	}

	Matrix<T,COLMAJOR> operator+(Matrix<T,COLMAJOR> B)
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Matrix: Addition cannot be performed due to incompatible sizes\n";
			Matrix<T,COLMAJOR> C(1,1);
			return C;
		}
#endif
		Matrix<T,COLMAJOR> C(nrows, ncols);
		int i;

		for(i = 0; i < C.size; i++)
			C.elems[i] = elems[i] + B.elems[i];
		return C;
	}

	Matrix<T,COLMAJOR> operator-(Matrix<T,COLMAJOR> B)
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Matrix: Subtraction cannot be performed due to incompatible sizes\n";
			Matrix<T,COLMAJOR> C(1,1);
			return C;
		}
#endif
		Matrix<T,COLMAJOR> C(nrows, ncols);

		for(int i = 0; i < C.size; i++)
			C.elems[i] = elems[i] - B.elems[i];
		return C;
	}

	Matrix<T,COLMAJOR> operator*(Matrix<T,COLMAJOR> B)
	{
		Matrix<double,COLMAJOR> C(nrows, B.cols());
		C.zeros();
#ifdef DEBUG
		if(ncols != B.rows())
		{
			std::cout << "! Matrix: Multiplication cannot be performed - incompatible sizes!\n";
			return C;
		}
#endif
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < B.cols(); j++)
				for(int k = 0; k < ncols; k++)
					C(i,j) += get(i,k) * B.get(k,j);

		return C;
	}

	T dot_product(const Matrix<T,COLMAJOR>& A)
	/* Returns sum of products of respective elements of flattened arrays containing matrix elements of this and A */
	{
		T* elemsA = A.elems;
		#ifdef _OPENMP
		T* elems = this->elems;
		int size = this->size;
		#endif
		int i;
		double ans = 0;
		#pragma omp parallel for if(size >= 4) default(none) private(i) shared(elems,elemsA,size) reduction(+: ans) num_threads(nthreads_m)
		for(i = 0; i < size; i++)
		{
			T temp = elems[i]*elemsA[i];
			//#pragma omp critical (omp_dot)
			ans += temp;
		}
		return ans;
	}
};

} //end namespace amat
