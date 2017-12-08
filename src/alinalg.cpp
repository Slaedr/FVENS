#include "alinalg.hpp"
#include <iostream>
#include <vector>
#include <cstring>

#define PETSCOPTION_STR_LEN 30

namespace acfd {

template <int nvars>
StatusCode setupMatrix(const UMesh2dh *const m, Mat *const A)
{
	StatusCode ierr = 0;
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);

	ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, m->gnelem()*nvars, m->gnelem()*nvars); 
	CHKERRQ(ierr);
	ierr = MatSetBlockSize(*A, nvars); CHKERRQ(ierr);
	
	std::vector<PetscInt> dnnz(m->gnelem());
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		dnnz[iel] = m->gnfael(iel);
	}
	ierr = MatSeqBAIJSetPreallocation(*A, nvars, 0, &dnnz[0]); CHKERRQ(ierr);
	ierr = MatMPIBAIJSetPreallocation(*A, nvars, 0, &dnnz[0], 1, NULL); CHKERRQ(ierr);

	dnnz.resize(m->gnelem()*nvars);
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++) {
			dnnz[iel*nvars+i] = m->gnfael(iel)*nvars;
		}
	}
	
	ierr = MatSeqAIJSetPreallocation(*A, 0, &dnnz[0]); CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(*A, 0, &dnnz[0], nvars, NULL); CHKERRQ(ierr);

	return ierr;
}

template PetscErrorCode setupMatrixStorage<NVARS>(const UMesh2dh *const m, Mat *const A);
template PetscErrorCode setupMatrixStorage<1>(const UMesh2dh *const m, Mat *const A);

StatusCode setupVectors(const Mat A, Vec *const u, Vec *const r)
{
	StatusCode ierr = 0;
	ierr = MatCreateVecs(A, u, r); CHKERRQ(ierr);
	return ierr;
}

}
