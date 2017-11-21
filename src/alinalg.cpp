#include "alinalg.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>

template <short nvars>
PetscErrorCode setupMatrix(const UMesh2dh *const m, Mat A)
{
	PetscErrorCode ierr = 0;
	MatCreate(PETSC_COMM_WORLD,&A);
	ierr = MatSetFromOptions(A); CHKERRQ(ierr);

	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m->gnelem()*nvars, m->gnelem()*nvars);
	std::vector<PetscInt> dnnz(m->gnelem()*nvars);
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++) {
			dnnz[iel*nvars+i] = m->gnfael(iel)*nvars;
		}
	}

	ierr = MatMPIAIJSetPreallocation(A, 3*nvars, &dnnz[0], nvars, NULL); CHKERRQ(ierr);

	dnnz.resize(m->gnelem());
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		dnnz[iel] = m->gnfael(iel);
	}
	
	ierr = MatMPIBAIJSetPreallocation(A, 3, &dnnz[0], 1, NULL); CHKERRQ(ierr);

	/*MatType mattype; 
	ierr = MatGetType(A, &mattype); CHKERRQ(ierr);
	if(!std::strcmp(mattype,MATMPIAIJ)) {
		// scalar AIJ format	
	}
	else if(mattype == MATMPIBAIJ) {
		// construct non-zero structure for block sparse format

	}*/

	return ierr;
}

template PetscErrorCode setupMatrixStorage<NVARS>(const UMesh2dh *const m, Mat A);
template PetscErrorCode setupMatrixStorage<1>(const UMesh2dh *const m, Mat A);

}
