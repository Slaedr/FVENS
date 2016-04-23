#include <amatrix.hpp>

namespace amat {

/// Recursively computes the determinant of a matrix.
/** Note that the algorithm used here is inefficient - by my estimate, it takes O(n!) operations to compute, where n is the size of the matrix.
* However, for small matrices, say n < 6, this naive method is probably better than some other method such as LU decomposition. The latter is O(n^3).
* \note NOTE: This function does not work (as of 8 February 2016)!!
*/
template <typename T>
T determinant(const Matrix<T>& mat)
{
#ifdef DEBUG
	if(mat.nrows != mat.ncols || mat.nrows < 2) {
		std::cout << "Matrix: determinant(): Size error!" << std::endl;
		return 0;
	}
#endif
	if(mat.nrows == 2)
		return mat.get(0,0)*mat.get(1,1) - mat.get(0,1)*mat.get(1,0);
	else
	{
		T det;
		// create minors
		Matrix<T>* submat;
		submat = new Matrix<T>[mat.nrows];
		T* dets = new T[mat.nrows];
		for(acfd_int k = 0; k < mat.nrows; k++)
		{
			submat[k].setup(mat.nrows-1,mat.nrows-1);
			acfd_int i,j, ii=0, jj=0;
			for(i = 1; i < mat.nrows; i++)		// leave first row
			{
				jj = 0;
				for(j = 0; j < mat.ncols; j++)
				{
					if(j == k) continue;
					submat[k](ii,jj) = mat.get(i,j);
					//std::cout << "Element " << i << "," << j << " of original matrix stored in element " << ii << "," << jj << " of sub-matrix " << k << std::endl;
					jj++;
				}
				ii++;
			}
			// recursive call
			dets[k] = determinant(submat[k]);
			// compute the determinant
			det += (T)(pow(-1.0,k+1))*mat.get(0,k)*dets[k];
		}
		delete [] submat;
		delete [] dets;
		return det;
	}
}

}
