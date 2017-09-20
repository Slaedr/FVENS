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
 * \brief Includes array storage classes.
 */

#ifndef __AARRAY2D_H
#define __AARRAY2D_H

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

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
 * If A is a column-major matrix, A[i][j] == A[j * nrows + i] where i is the row-index and j is the column index.
 */
template <class T>
class Array2d
{
private:
	a_int nrows;
	a_int ncols;
	a_int size;
	T* elems;
	bool isalloc;

public:
	///No-arg constructor. Note: no memory allocation! Make sure Array2d::setup(a_int,a_int,MStype) is used.
	Array2d()
	{
		nrows = 0; ncols = 0; size = 0;
		isalloc = false;
	}

	// Full-arg constructor
	Array2d(a_int nr, a_int nc)
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

	Array2d(const Array2d<T>& other)
	{
		nrows = other.nrows;
		ncols = other.ncols;
		size = nrows*ncols;
		elems = new T[nrows*ncols];
		isalloc = true;
		for(a_int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = other.elems[i];
		}
	}

	~Array2d()
	{
		if(isalloc == true)	
			delete [] elems;
		isalloc = false;
	}

	Array2d<T>& operator=(const Array2d<T>& rhs)
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
		for(a_int i = 0; i < nrows*ncols; i++)
		{
			elems[i] = rhs.elems[i];
		}
		return *this;
	}

	/// Separate setup function in case no-arg constructor has to be used
	/** \deprecated Please use resize() instead.
	 */
	void setup(const a_int nr, const a_int nc)
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
		if(isalloc == true)
			delete [] elems;
		elems = new T[nrows*ncols];
		isalloc = true;
	}
	
	/// Sets a new size for the array, deletes the contents and allocates new memory
	void resize(const a_int nr, const a_int nc)
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
		if(isalloc == true)
			delete [] elems;
		elems = new T[nrows*ncols];
		isalloc = true;
	}

	/// Setup without deleting earlier allocation: use in case of Array2d<t>* (pointer to Array2d<t>)
	void setupraw(a_int nr, a_int nc)
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
		for(a_int i = 0; i < size; i++)
			elems[i] = (T)(0.0);
	}

	void ones()
	{
		for(a_int i = 0; i < size; i++)
			elems[i] = 1;
	}

	void identity()
	{
		T one = (T)(1);
		T zero = (T)(0);
		for(a_int i = 0; i < nrows; i++)
			for(a_int j = 0; j < ncols; j++)
				if(i==j) operator()(i,j) = one;
				else operator()(i,j) = zero;
	}

	/// function to set matrix elements from a ROW-MAJOR array
	void setdata(const T* A, a_int sz)
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

	T get(const a_int i, const a_int j=0) const
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Array2d: get(): Index beyond array size(s)\n"; return 0; }
		if(i < 0 || j < 0) { std::cout << "Array2d: get(): Index less than 0!\n"; return 0; }
#endif
		return elems[i*ncols + j];
	}

	void set(a_int i, a_int j, T data)
	{
#ifdef DEBUG
		if(i>=nrows || j>=ncols) { std::cout << "Array2d: set(): Index beyond array size(s)\n"; return; }
		if(i < 0 || j < 0) {std::cout << "Array2d: set(): Negative index!\n"; return; }
#endif
		elems[i*ncols + j] = data;
	}

	a_int rows() const { return nrows; }
	a_int cols() const { return ncols; }
	a_int msize() const { return size; }

	/// Prints the matrix to standard output.
	void mprint() const
	{
		std::cout << "\n";
		for(a_int i = 0; i < nrows; i++)
		{
			for(a_int j = 0; j < ncols; j++)
				std::cout << std::setw(WIDTH) << std::setprecision(WIDTH/2+1) << elems[i*ncols+j];
			std::cout << std::endl;
		}
	}

	/// Prints the matrix to file
	void fprint(std::ofstream& outfile) const
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

	/// Reads matrix from file
	void fread(std::ifstream& infile)
	{
		infile >> nrows; infile >> ncols;
		size = nrows*ncols;
		delete [] elems;
		elems = new T[nrows*ncols];
		for(a_int i = 0; i < nrows; i++)
			for(a_int j = 0; j < ncols; j++)
				infile >> elems[i*ncols + j];
	}

	/// Getter/setter function for expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	T& operator()(const a_int x, const a_int y=0)
	{
#ifdef DEBUG
		if(x>=nrows || y>=ncols) { std::cout << "! Array2d (): Index beyond array size(s)\n"; /*return elems[0];*/ }
		if(x<0 || y<0) { std::cout << "! Array2d (): Index negative!\n"; /*return elems[0];*/ }
#endif
		return elems[x*ncols + y];
	}
	
	/// Const Getter/setter function for expressions like A(1,2) = 141 to set the element at 1st row and 2nd column to 141
	const T& operator()(const a_int x, const a_int y=0) const
	{
#ifdef DEBUG
		if(x>=nrows || y>=ncols) { std::cout << "Array2d (): Index beyond array size(s)\n"; /*return elems[0];*/ }
		if(x<0 || y<0) { std::cout << "! Array2d (): Index negative!\n"; /*return elems[0];*/ }
#endif
		return elems[x*ncols + y];
	}

	/// Returns a pointer-to-const to the beginning of a row
	const T* const_row_pointer(const a_int r) const
	{
#ifdef DEBUG
		if(r >= nrows) { std::cout << "! Array2d: const_row_pointer(): Row index beyond array size!\n"; return nullptr;}
#endif
		return &elems[r*ncols];
	}
	
	/// Returns a pointer to the beginning of a row
	T* row_pointer(const a_int r)
	{
#ifdef DEBUG
		if(r >= nrows) { std::cout << "! Array2d: row_pointer(): Row index beyond array size!\n"; return nullptr;}
#endif
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

	T minincol(a_int j) const
	{
		T min = get(0,j);
		for(a_int i = 0; i < nrows; i++)
			if(min > get(i,j)) min = get(i,j);
		return min;
	}

	T mininrow(a_int i) const
	{
		T max = get(i,0);
		for(a_int j = 0; j < nrows; j++)
			if(max > get(i,j)) max = get(i,j);
		return max;
	}

	T min() const
	{
		T max = elems[0];
		for(a_int i = 0; i < size; i++)
			if(elems[i] < max) max = elems[i];
		return max;
	}

	T average() const
	{
		T avg = 0;
		for(a_int i = 0; i < size; i++)
			avg += elems[i];
		avg = avg/size;
		return avg;
	}

	// sums the square of all elements in the matrix and returns the square root of this sum
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

	/// function to return a sub-matrix of this matrix
	Array2d<T> sub(a_int startr, a_int startc, a_int offr, a_int offc) const
	{
		Array2d<T> B(offr, offc);
		for(a_int i = 0; i < offr; i++)
			for(a_int j = 0; j < offc; j++)
				B(i,j) = elems[(startr+i)*ncols + startc + j];
		return B;
	}

	/// Function that returns a given column of the matrix as a row-major matrix
	Array2d<T> col(a_int j) const
	{
		Array2d<T> b(nrows, 1);
		for(a_int i = 0; i < nrows; i++)
			b(i,0) = elems[i*ncols + j];
		return b;
	}

	Array2d<T> row(a_int i) const
	{
		Array2d<T> b(1, ncols);
		for(a_int j = 0; j < ncols; j++)
			b(0,j) = elems[i*ncols + j];
		return b;
	}

	/*//Function to return a reference to a given column of the matrix
	Array2d<T>& colr(a_int j)
	{
		//Array2d<T>* b(nrows, 1);
		Array2d<T>* b; b->elems.reserve(nrows);
		for(a_int i = 0; i < nrows; i++)
			b.elems[i] = &elems[i*ncols + j];
		return *b;
	} */

	/// Function for replacing a column of the matrix with a vector. NOTE: No check for whether b is really a vector - which it must be.
	void replacecol(a_int j, Array2d<T> b)
	{
#ifdef DEBUG
		if(b.cols() != 1 || b.rows() != nrows) { std::cout << "\nSize error in replacecol"; return; }
#endif
		for(a_int i = 0; i < nrows; i++)
			elems[i*ncols + j] = b.elems[i];
	}

	/// Function for replacing a row
	void replacerow(a_int i, Array2d<T> b)
	{
#ifdef DEBUG
		if(b.cols() != ncols || b.rows() != 1) { std::cout << "\nSize error in replacerow"; return; }
#endif
		for(a_int j = 0; j < ncols; j++)
			elems[i*ncols + j] = b.elems[j];
	}

	/// Returns the transpose
	Array2d<T> trans() const
	{
		Array2d<T> t(ncols, nrows);
		for(a_int i = 0; i < ncols; i++)
			for(a_int j = 0; j < nrows; j++)
				t(i,j) = get(j,i);
		return t;
	}

	/// Multiply a matrix by a scalar. Note: only expressions of type A*3 work, not 3*A
	Array2d<T> operator*(T num)
	{
		Array2d<T> A(nrows,ncols);
		a_int i;

		for(i = 0; i < A.size; i++)
			A.elems[i] = elems[i] * num;
		return A;
	}

	/**	The matrix addition and subtraction operators are inefficient! Do not use in long loops. */
	Array2d<T> operator+(Array2d<T> B) const
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Array2d: Addition cannot be performed due to incompatible sizes\n";
			Array2d<T> C(1,1);
			return C;
		}
#endif
		Array2d<T> C(nrows, ncols);
		a_int i;

		for(i = 0; i < C.size; i++)
			C.elems[i] = elems[i] + B.elems[i];
		return C;
	}

	Array2d<T> operator-(Array2d<T> B) const
	{
#ifdef DEBUG
		if(nrows != B.rows() || ncols != B.cols())
		{
			std::cout << "! Array2d: Subtraction cannot be performed due to incompatible sizes\n";
			Array2d<T> C(1,1);
			return C;
		}
#endif
		Array2d<T> C(nrows, ncols);

		for(a_int i = 0; i < C.size; i++)
			C.elems[i] = elems[i] - B.elems[i];
		return C;
	}

	Array2d<T> operator*(Array2d<T> B)
	{
		Array2d<a_real> C(nrows, B.cols());
		C.zeros();
#ifdef DEBUG
		if(ncols != B.rows())
		{
			std::cout << "! Array2d: Multiplication cannot be performed - incompatible sizes!\n";
			return C;
		}
#endif
		for(a_int i = 0; i < nrows; i++)
			for(a_int j = 0; j < B.cols(); j++)
				for(a_int k = 0; k < ncols; k++)
					C(i,j) += get(i,k) * B.get(k,j);
					//C.set( C.get(i,j) + get(i,k)*B.get(k,j), i,j );

		return C;
	}

	/// Returns sum of products of respective elements of flattened arrays containing matrix elements of this and A
	T dot_product(const Array2d<T>& A)
	{
		T* elemsA = A.elems;
		#ifdef _OPENMP
		T* elems = this->elems;
		a_int size = this->size;
		#endif
		a_int i;
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
	}
};


} //end namespace amat

#endif
