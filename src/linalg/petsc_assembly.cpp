/** \file
 * \brief Implementation of PETSc assembly functions
 * \date 2018-10
 */

#include "petsc_assembly.hpp"

namespace fvens {

template <typename scalar, int nvars>
StatusCode assemble_residual(const Spatial<scalar,nvars> *const spatial,
                             const Vec uvec, Vec __restrict rvec,
                             const bool gettimesteps, Vec __restrict dtmvec)
{
	StatusCode ierr = 0;

	const UMesh2dh<scalar> *const m = spatial->mesh();

	amat::Array2d<a_real> integ, ug, uleft, uright;
	integ.resize(m->gnelem(), 1);
	ug.resize(m->gnbface(),nvars);
	uleft.resize(m->gnaface(), nvars);
	uright.resize(m->gnaface(), nvars);

	PetscInt locnelem, dtsz;
	const PetscScalar *uarr;
	PetscScalar *rarr, *dtm = NULL;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);

	if(gettimesteps) {
		ierr = VecGetLocalSize(dtmvec, &dtsz); CHKERRQ(ierr);
		assert(locnelem == dtsz);
		ierr = VecGetArray(dtmvec, &dtm); CHKERRQ(ierr);
	}

	spatial->compute_residual(uarr, rarr, gettimesteps, dtm);

	VecRestoreArrayRead(uvec, &uarr);
	VecRestoreArray(rvec, &rarr);
	VecRestoreArray(dtmvec, &dtm);
	return ierr;
}

template <typename scalar, int nvars>
StatusCode assemble_jacobian(const Spatial<scalar,nvars> *const spatial, const Vec uvec, Mat A)
{
	StatusCode ierr = 0;

	const UMesh2dh<scalar> *const m = spatial->mesh();

	PetscInt locnelem; const PetscScalar *uarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);

		Matrix<a_real,nvars,nvars,RowMajor> left;
		spatial->compute_local_jacobian_boundary(iface, &uarr[lelem*nvars], left);

		// negative L and U contribute to diagonal blocks
		left *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1,&lelem, 1,&lelem, left.data(), ADD_VALUES);
		}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);

		Matrix<a_real,nvars,nvars,RowMajor> L;
		Matrix<a_real,nvars,nvars,RowMajor> U;
		spatial->compute_local_jacobian_interior(iface, &uarr[lelem*nvars], &uarr[relem*nvars], L, U);

#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relem, 1, &lelem, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &relem, U.data(), ADD_VALUES);
		}

		// negative L and U contribute to diagonal blocks
		L *= -1.0; U *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &lelem, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relem, 1, &relem, U.data(), ADD_VALUES);
		}
	}

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);

	return ierr;
}

template StatusCode
assemble_residual<a_real,1>(const Spatial<a_real,1> *const spatial,
                            const Vec uvec, Vec __restrict rvec,
                            const bool gettimesteps, Vec dtm);
template StatusCode
assemble_residual<a_real,NVARS>(const Spatial<a_real,NVARS> *const spatial,
                                const Vec uvec, Vec __restrict rvec,
                                const bool gettimesteps, Vec dtm);

template StatusCode
assemble_jacobian<a_real,1>(const Spatial<a_real,1> *const spatial, const Vec uvec, Mat A);
template StatusCode
assemble_jacobian<a_real,NVARS>(const Spatial<a_real,NVARS> *const spatial, const Vec uvec, Mat A);

}
