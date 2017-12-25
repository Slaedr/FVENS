#include "alinalg.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>

#define PETSCOPTION_STR_LEN 30

namespace acfd {

template<int nvars>
StatusCode setupMatrixFreeJacobian(MatrixFreeSpatialJacobian *const mfjac, Mat A)
{
	// TODO: setup matrix-free Jacobian
}

template <int nvars>
StatusCode setupSystemMatrix(const UMesh2dh *const m, Mat *const A)
{
	StatusCode ierr = 0;
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = MatSetType(*A, MATMPIBAIJ); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);

	ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, m->gnelem()*nvars, m->gnelem()*nvars); 
	CHKERRQ(ierr);
	ierr = MatSetBlockSize(*A, nvars); CHKERRQ(ierr);
	
	std::vector<PetscInt> dnnz(m->gnelem());
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		dnnz[iel] = m->gnfael(iel)+1;
	}
	ierr = MatSeqBAIJSetPreallocation(*A, nvars, 0, &dnnz[0]); CHKERRQ(ierr);
	ierr = MatMPIBAIJSetPreallocation(*A, nvars, 0, &dnnz[0], 1, NULL); CHKERRQ(ierr);

	dnnz.resize(m->gnelem()*nvars);
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++) {
			dnnz[iel*nvars+i] = (m->gnfael(iel)+1)*nvars;
		}
	}
	
	ierr = MatSeqAIJSetPreallocation(*A, 0, &dnnz[0]); CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(*A, 0, &dnnz[0], nvars, NULL); CHKERRQ(ierr);

	ierr = MatSetOption(*A, MAT_USE_HASH_TABLE, PETSC_TRUE); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setupSystemMatrix<NVARS>(const UMesh2dh *const m, Mat *const A);
template StatusCode setupSystemMatrix<1>(const UMesh2dh *const m, Mat *const A);

template<int nvars>
MatrixFreeSpatialJacobian<nvars>::MatrixFreeSpatialJacobian(const Spatial<nvars> *const space, 
		const a_real epsilon)
	: spatial{space}, eps{epsilon}
{ }

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::setup_work_storage(const Vec system_vector)
{
	StatusCode ierr = VecDuplicate(system_vector, &aux); CHKERRQ(ierr);
	return ierr;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::destroy_work_storage()
{
	StatusCode ierr = VecDestroy(&aux); CHKERRQ(ierr);
	return ierr;
}

template<int nvars>
void MatrixFreeSpatialJacobian<nvars>::set_state(const Vec u_state, const Vec r_state) {
	u = u_state;
	res = r_state;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::apply(const Vec x, Vec y) const
{
	StatusCode ierr = 0;
	std::vector<a_real> dummy;

	PetscScalar xnorm = 0;
	ierr = VecNorm(x, NORM_2, &xnorm); CHKERRQ(ierr);
#ifdef DEBUG
	if(xnorm < 10*std::numeric_limits<a_real>::epsilon)
		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP, 
				"Norm of offset is too small for finite difference Jacobian!");
#endif
	xnorm = eps/xnorm;
	ierr = VecAXPBY(aux, xnorm, 0.0, x); CHKERRQ(ierr);
	ierr = VecAXPY(aux, 1.0, u); CHKERRQ(ierr);
	ierr = spatial->compute_residual(aux, y, false, dummy); CHKERRQ(ierr);
	ierr = VecAXPY(y, -1.0, res); CHKERRQ(ierr);
	ierr = VecScale(y, xnorm); CHKERRQ(ierr);
	
	return ierr;
}

template class MatrixFreeSpatialJacobian<NVARS>;
template class MatrixFreeSpatialJacobian<1>;

template <int nvars>
StatusCode setup_matrixfree_jacobian(MatrixFreeSpatialJacobian<nvars> *const mfj, Mat A)
{
	// TODO
}

template StatusCode setup_matrixfree_jacobian<NVARS>(MatrixFreeSpatialJacobian<NVARS> *const mfj, 
		Mat A);
template StatusCode setup_matrixfree_jacobian<1>(MatrixFreeSpatialJacobian<1> *const mfj, 
		Mat A);

}
