#include "alinalg.hpp"
#include <iostream>
#include <vector>
#include <cstring>

#define PETSCOPTION_STR_LEN 30

namespace acfd {

template <int nvars>
StatusCode setupMatrix(const UMesh2dh *const m, Mat A)
{
	StatusCode ierr = 0;
	MatCreate(PETSC_COMM_WORLD,&A);
	ierr = MatSetFromOptions(A); CHKERRQ(ierr);

	ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m->gnelem()*nvars, m->gnelem()*nvars); 
	CHKERRQ(ierr);
	ierr = MatSetBlockSize(A, nvars); CHKERRQ(ierr);
	
	PetscBool set = PETSC_FALSE;
	PetscOptionsHasName(NULL, NULL, "-mat_type", &set);
	char matstr[PETSCOPTION_STR_LEN];
	PetscBool flag = PETSC_FALSE;
	PetscOptionsGetString(NULL, NULL, "-mat_type", matstr, PETSCOPTION_STR_LEN, &flag);

	if(set == PETSC_FALSE || flag == PETSC_FALSE || 
			!std::strcomp(matstr,"baij") || !std::strcomp(matstr,"seqbaij") || 
			!std::strcomp(matstr,"mpibaij"))
	{
		std::cout << "setupMatrix: Setting default of MPI block matrix.\n";
		ierr = MatSetType(A, MATMPIBAIJ); CHKERRQ(ierr);

		std::vector<PetscInt> dnnz(m->gnelem());
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			dnnz[iel] = m->gnfael(iel);
		}
		ierr = MatSeqBAIJSetPreallocation(A, nvars, 3, &dnnz[0]); CHKERRQ(ierr);
		ierr = MatMPIBAIJSetPreallocation(A, nvars, 3, &dnnz[0], 1, NULL); CHKERRQ(ierr);
	}
	else {	
		std::cout << "setupMatrix: Setting MPI scalar matrix.\n";
		ierr = MatSetType(A, MATMPIAIJ); CHKERRQ(ierr);

		std::vector<PetscInt> dnnz(m->gnelem()*nvars);
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++) {
				dnnz[iel*nvars+i] = m->gnfael(iel)*nvars;
			}
		}
		
		ierr = MatSeqAIJSetPreallocation(A, 3*nvars, &dnnz[0], nvars, NULL); CHKERRQ(ierr);
		ierr = MatMPIAIJSetPreallocation(A, 3*nvars, &dnnz[0], nvars, NULL); CHKERRQ(ierr);
	}


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
