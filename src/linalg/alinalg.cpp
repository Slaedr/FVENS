#include "alinalg.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>

namespace fvens {

template <int nvars>
static StatusCode setJacobianSizes(const UMesh2dh<a_real> *const m, Mat A) 
{
	StatusCode ierr = 0;
	ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m->gnelem()*nvars, m->gnelem()*nvars);
	CHKERRQ(ierr);
	ierr = MatSetBlockSize(A, nvars); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
StatusCode setJacobianPreallocation(const UMesh2dh<a_real> *const m, Mat A) 
{
	// The implementation must be changed for the multi-process case
	
	StatusCode ierr = 0;

	// set block preallocation
	std::vector<PetscInt> dnnz(m->gnelem());
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		dnnz[iel] = m->gnfael(iel)+1;
	}
	ierr = MatSeqBAIJSetPreallocation(A, nvars, 0, &dnnz[0]); CHKERRQ(ierr);
	ierr = MatMPIBAIJSetPreallocation(A, nvars, 0, &dnnz[0], 1, NULL); CHKERRQ(ierr);

	// set scalar (non-block) preallocation
	dnnz.resize(m->gnelem()*nvars);
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++) {
			dnnz[iel*nvars+i] = (m->gnfael(iel)+1)*nvars;
		}
	}

	ierr = MatSeqAIJSetPreallocation(A, 0, &dnnz[0]); CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(A, 0, &dnnz[0], nvars, NULL); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setJacobianPreallocation<1>(const UMesh2dh<a_real> *const m, Mat A);

template <int nvars>
StatusCode setupSystemMatrix(const UMesh2dh<a_real> *const m, Mat *const A)
{
	StatusCode ierr = 0;
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = MatSetType(*A, MATMPIBAIJ); CHKERRQ(ierr);

	ierr = setJacobianSizes<nvars>(m, *A); CHKERRQ(ierr);

	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);

	ierr = setJacobianPreallocation<nvars>(m, *A); CHKERRQ(ierr);

	ierr = MatSetUp(*A); CHKERRQ(ierr);

	ierr = MatSetOption(*A, MAT_USE_HASH_TABLE, PETSC_TRUE); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setupSystemMatrix<NVARS>(const UMesh2dh<a_real> *const m, Mat *const A);
template StatusCode setupSystemMatrix<1>(const UMesh2dh<a_real> *const m, Mat *const A);

template<int nvars>
MatrixFreeSpatialJacobian<nvars>::MatrixFreeSpatialJacobian()
	: eps{1e-7}
{
	PetscBool set = PETSC_FALSE;
	PetscOptionsGetReal(NULL, NULL, "-matrix_free_difference_step", &eps, &set);
}

template<int nvars>
void MatrixFreeSpatialJacobian<nvars>::set_spatial(const Spatial<nvars> *const space) {
	spatial = space;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::setup_work_storage(const Mat system_matrix)
{
	StatusCode ierr = MatCreateVecs(system_matrix, NULL, &aux); CHKERRQ(ierr);
	ierr = VecSet(aux,0.0); CHKERRQ(ierr);
	std::cout << " MatrixFreeSpatialJacobian: Using finite difference step " << eps << '\n';
	return ierr;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::destroy_work_storage()
{
	StatusCode ierr = VecDestroy(&aux); CHKERRQ(ierr);
	return ierr;
}

template<int nvars>
void MatrixFreeSpatialJacobian<nvars>::set_state(const Vec u_state, const Vec r_state,
		const std::vector<a_real> *const dtms) 
{
	u = u_state;
	res = r_state;
	mdt = dtms;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::apply(const Vec x, Vec y) const
{
	StatusCode ierr = 0;
	std::vector<a_real> dummy;
	const UMesh2dh<a_real> *const m = spatial->mesh();
	ierr = VecSet(y, 0.0); CHKERRQ(ierr);

	const a_real *xr;
	a_real *yr;
	ierr = VecGetArray(y, &yr); CHKERRQ(ierr);
	ierr = VecGetArrayRead(x, &xr); CHKERRQ(ierr);

	PetscScalar xnorm = 0;
	ierr = VecNorm(x, NORM_2, &xnorm); CHKERRQ(ierr);
#ifdef DEBUG
	if(xnorm < 10.0*std::numeric_limits<a_real>::epsilon())
		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP,
				"Norm of offset is too small for finite difference Jacobian!");
#endif
	xnorm = eps/xnorm;

	// aux <- eps/xnorm * x
	ierr = VecAXPBY(aux, xnorm, 0.0, x); CHKERRQ(ierr);
	// aux <- u + eps/xnorm * x
	ierr = VecAXPY(aux, 1.0, u); CHKERRQ(ierr);
	// y <- -r(u + eps/xnorm * x)
	ierr = spatial->compute_residual(aux, y, false, dummy); CHKERRQ(ierr);
	// y <- -(-r(u + eps/xnorm * x)) + (-r(u)) = r(u + eps/xnorm * x) - r(u)
	ierr = VecAXPBY(y, 1.0, -1.0, res); CHKERRQ(ierr);
	
	/* divide by the normalized step length */
	ierr = VecScale(y, 1.0/xnorm); CHKERRQ(ierr);

	// finally, add the pseudo-time term (Vol/dt du = Vol/dt x)
#pragma omp parallel for simd default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++)
			yr[iel*nvars+i] += (*mdt)[iel] * xr[iel*nvars+i];
	}
	
	ierr = VecRestoreArray(y, &yr); CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(x, &xr); CHKERRQ(ierr);
	return ierr;
}

template class MatrixFreeSpatialJacobian<NVARS>;
template class MatrixFreeSpatialJacobian<1>;

/// The function called by PETSc to carry out a Jacobian-vector product
template <int nvars>
StatusCode matrixfree_apply(Mat A, Vec x, Vec y)
{
	StatusCode ierr = 0;
	MatrixFreeSpatialJacobian<nvars> *mfmat;
	ierr = MatShellGetContext(A, (void*)&mfmat); CHKERRQ(ierr);
	ierr = mfmat->apply(x,y); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
StatusCode matrixfree_destroy(Mat A)
{
	StatusCode ierr = 0;
	MatrixFreeSpatialJacobian<nvars> *mfmat;
	ierr = MatShellGetContext(A, (void*)&mfmat); CHKERRQ(ierr);
	ierr = mfmat->destroy_work_storage(); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
StatusCode setup_matrixfree_jacobian(const UMesh2dh<a_real> *const m,
		MatrixFreeSpatialJacobian<nvars> *const mfj, Mat *const A)
{
	StatusCode ierr = 0;
	
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = setJacobianSizes<nvars>(m, *A); CHKERRQ(ierr);
	ierr = MatSetType(*A, MATSHELL); CHKERRQ(ierr);

	ierr = mfj->setup_work_storage(*A); CHKERRQ(ierr);
	ierr = MatShellSetContext(*A, (void*)mfj); CHKERRQ(ierr);
	ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))&matrixfree_apply<nvars>); 
	CHKERRQ(ierr);
	ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))&matrixfree_destroy<nvars>);
	CHKERRQ(ierr);

	ierr = MatSetUp(*A); CHKERRQ(ierr);
	return ierr;
}

template StatusCode setup_matrixfree_jacobian<NVARS>( const UMesh2dh<a_real> *const m,
		MatrixFreeSpatialJacobian<NVARS> *const mfj,
		Mat *const A);
template StatusCode setup_matrixfree_jacobian<1>( const UMesh2dh<a_real> *const m,
		MatrixFreeSpatialJacobian<1> *const mfj,
		Mat *const A);

bool isMatrixFree(Mat M) 
{
	MatType mattype;
	StatusCode ierr = MatGetType(M, &mattype);
	if(ierr != 0)
		throw "Could not get matrix type!";

	if(!strcmp(mattype,"shell"))
		return true;
	else
		return false;
}

/// Recursive function to return the first occurrence if a specific type of PC
StatusCode getPC(KSP ksp, const char *const type_name, PC* pcfound)
{
	StatusCode ierr = 0;
	PC pc;
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
	PetscBool isbjacobi, isasm, ismg, isgamg, isksp, isrequired;
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&isbjacobi); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&isasm); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&ismg); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCGAMG,&isgamg); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCKSP,&isksp); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,type_name,&isrequired); CHKERRQ(ierr);

	if(isrequired) {
		// base case
		*pcfound = pc;
	}
	else if(isbjacobi || isasm)
	{
		PetscInt nlocalblocks, firstlocalblock;
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP *subksp;
		if(isbjacobi) {
			ierr = PCBJacobiGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp); CHKERRQ(ierr);
		}
		else {
			ierr = PCASMGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp); CHKERRQ(ierr);
		}
		if(nlocalblocks != 1)
			SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, 
					"Only one subdomain per rank is supported.");
		ierr = getPC(subksp[0], type_name, pcfound); CHKERRQ(ierr);
	}
	else if(ismg || isgamg) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		PetscInt nlevels;
		ierr = PCMGGetLevels(pc, &nlevels); CHKERRQ(ierr);
		for(int ilvl = 0; ilvl < nlevels; ilvl++) {
			KSP smootherctx;
			ierr = PCMGGetSmoother(pc, ilvl , &smootherctx); CHKERRQ(ierr);
			ierr = getPC(smootherctx, type_name, pcfound); CHKERRQ(ierr);
		}
		KSP coarsesolver;
		ierr = PCMGGetCoarseSolve(pc, &coarsesolver); CHKERRQ(ierr);
		ierr = getPC(coarsesolver, type_name, pcfound); CHKERRQ(ierr);
	}
	else if(isksp) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP subksp;
		ierr = PCKSPGetKSP(pc, &subksp); CHKERRQ(ierr);
		ierr = getPC(subksp, type_name, pcfound); CHKERRQ(ierr);
	}

	return ierr;
}

#ifdef USE_BLASTED

template <int nvars>
StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<nvars> *const startprob, Blasted_data_list& bctx)
{
	StatusCode ierr = 0;
	Mat M, A;
	ierr = KSPGetOperators(ksp, &A, &M); CHKERRQ(ierr);

	// first assemble the matrix once because PETSc requires it
	ierr = startprob->compute_jacobian(u, M); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = setup_blasted_stack(ksp,&bctx); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<NVARS> *const startprob, 
		Blasted_data_list& bctx);
template StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<1> *const startprob, 
		Blasted_data_list& bctx);
#endif



}
