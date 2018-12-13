#include "alinalg.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>
#include "petsc_assembly.hpp"

namespace fvens {

StatusCode createSystemVector(const UMesh2dh<a_real> *const m, const int nvars, Vec *const v)
{
	StatusCode ierr = VecCreateMPI(PETSC_COMM_WORLD, m->gnelem()*nvars, m->gnelemglobal()*nvars, v);
	CHKERRQ(ierr);
	ierr = VecSetFromOptions(*v); CHKERRQ(ierr);
	return ierr;
}

StatusCode createGhostedSystemVector(const UMesh2dh<a_real> *const m, const int nvars, Vec *const v)
{
	StatusCode ierr = 0;

	const std::vector<a_int> globindices = m->getConnectivityGlobalIndices();

	ierr = VecCreateGhostBlock(PETSC_COMM_WORLD, nvars, m->gnelem()*nvars,
	                           m->gnelemglobal()*nvars, m->gnConnFace(),
	                           globindices.data(), v);
	CHKERRQ(ierr);
	ierr = VecSetFromOptions(*v); CHKERRQ(ierr);
	return ierr;
}

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
MatrixFreeSpatialJacobian<nvars>::MatrixFreeSpatialJacobian(const Spatial<a_real,nvars> *const s)
	: spatial{s}, eps{1e-7}
{
	PetscBool set = PETSC_FALSE;
	PetscOptionsGetReal(NULL, NULL, "-matrix_free_difference_step", &eps, &set);
}

template<int nvars>
void MatrixFreeSpatialJacobian<nvars>::set_state(const Vec u_state, const Vec r_state,
		const Vec dtms) 
{
	u = u_state;
	res = r_state;
	mdt = dtms;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::apply(const Vec x, Vec y) const
{
	StatusCode ierr = 0;
	Vec dummy = NULL;

	if(!spatial)
		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_POINTER,
		        "Spatial context not set!");

	const UMesh2dh<a_real> *const m = spatial->mesh();
	//ierr = VecSet(y, 0.0); CHKERRQ(ierr);

	Vec aux;
	ierr = VecDuplicate(x, &aux); CHKERRQ(ierr);

	const a_real *xr, *dtmr, *ur, *resr;
	a_real *yr, *auxr;
	ierr = VecGetArray(aux, &auxr); CHKERRQ(ierr);
	ierr = VecGetArray(y, &yr); CHKERRQ(ierr);
	ierr = VecGetArrayRead(x, &xr); CHKERRQ(ierr);
	ierr = VecGetArrayRead(u, &ur); CHKERRQ(ierr);

	PetscScalar xnorm = 0;
	ierr = VecNorm(x, NORM_2, &xnorm); CHKERRQ(ierr);

#ifdef DEBUG
	if(xnorm < 10.0*std::numeric_limits<a_real>::epsilon())
		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP,
				"Norm of offset is too small for finite difference Jacobian!");
#endif
	const a_real pertmag = eps/xnorm;

	// aux <- u + eps/xnorm * x ;    y <- 0
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < m->gnelem()*nvars; i++) {
		yr[i] = 0;
		auxr[i] = ur[i] + pertmag * xr[i];
	}

	ierr = VecRestoreArray(aux, &auxr); CHKERRQ(ierr);
	ierr = VecRestoreArray(y, &yr); CHKERRQ(ierr);

	// y <- -r(u + eps/xnorm * x)
	ierr = spatial->compute_residual(aux, y, false, dummy); CHKERRQ(ierr);

	ierr = VecGetArray(y, &yr); CHKERRQ(ierr);
	ierr = VecGetArray(aux, &auxr); CHKERRQ(ierr);
	ierr = VecGetArrayRead(res, &resr); CHKERRQ(ierr);
	ierr = VecGetArrayRead(mdt, &dtmr); CHKERRQ(ierr);

	// y <- vol/dt x + (-(-r(u + eps/xnorm * x)) + (-r(u))) / eps |x|
	//    = vol/dt x + (r(u + eps/xnorm * x) - r(u)) / eps |x|
	/* We need to divide the difference by the step length scaled by the norm of x.
	 * We do NOT divide by epsilon, because we want the product of the Jacobian and x, which is
	 * the directional derivative (in the direction of x) multiplied by the norm of x.
	 */
#pragma omp parallel for simd default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++) {
			// finally, add the pseudo-time term (Vol/dt du = Vol/dt x)
			yr[iel*nvars+i] = dtmr[iel]*xr[iel*nvars+i] + (-yr[iel*nvars+i] + resr[iel*nvars+i])/pertmag;
		}
	}
	
	ierr = VecRestoreArray(y, &yr); CHKERRQ(ierr);
	ierr = VecRestoreArray(aux, &auxr); CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(x, &xr); CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(mdt, &dtmr); CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(u, &ur); CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(res, &resr); CHKERRQ(ierr);
	ierr = VecDestroy(&aux); CHKERRQ(ierr);
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

/// Function called by PETSc to cleanup the matrix-free mat
template <int nvars>
StatusCode matrixfree_destroy(Mat A)
{
	StatusCode ierr = 0;
	MatrixFreeSpatialJacobian<nvars> *mfmat;
	ierr = MatShellGetContext(A, (void*)&mfmat); CHKERRQ(ierr);
	delete mfmat;
	return ierr;
}

template <int nvars>
StatusCode create_matrixfree_jacobian(const Spatial<a_real,nvars> *const s, Mat *const A)
{
	StatusCode ierr = 0;

	const UMesh2dh<a_real> *const m = s->mesh();
	MatrixFreeSpatialJacobian<nvars> *const mfj = new MatrixFreeSpatialJacobian<nvars>(s);
	
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = setJacobianSizes<nvars>(m, *A); CHKERRQ(ierr);
	ierr = MatSetType(*A, MATSHELL); CHKERRQ(ierr);

	ierr = MatShellSetContext(*A, (void*)mfj); CHKERRQ(ierr);
	ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))&matrixfree_apply<nvars>); 
	CHKERRQ(ierr);
	ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))&matrixfree_destroy<nvars>); 
	CHKERRQ(ierr);

	ierr = MatSetUp(*A); CHKERRQ(ierr);
	return ierr;
}

template
StatusCode create_matrixfree_jacobian<NVARS>(const Spatial<a_real,NVARS> *const s, Mat *const A);
template
StatusCode create_matrixfree_jacobian<1>(const Spatial<a_real,1> *const s, Mat *const A);

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
StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<a_real,nvars> *const startprob,
                         Blasted_data_list& bctx)
{
	StatusCode ierr = 0;
	Mat M, A;
	ierr = KSPGetOperators(ksp, &A, &M); CHKERRQ(ierr);

	// first assemble the matrix once because PETSc requires it
	ierr = assemble_jacobian(startprob, u, M); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = setup_blasted_stack(ksp,&bctx); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<a_real,NVARS> *const startprob, 
		Blasted_data_list& bctx);
template StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<a_real,1> *const startprob, 
		Blasted_data_list& bctx);
#endif



}
