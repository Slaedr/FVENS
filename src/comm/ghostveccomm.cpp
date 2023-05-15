/** \file
 * \brief Forward or reverse communication for ghosted vectors
 * \author Aditya Kashi
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ghostveccomm.hpp"
#include "utilities/aarray2d.hpp"
#include "linalg/petscutils.hpp"

namespace fvens {

template <int bs>
GhostedBlockVecComm<bs>::GhostedBlockVecComm(const UMesh<PetscReal,NDIM> *const mesh)
	: msh{mesh}, ltv(*msh), commUnderWay{false}
{ }

template <int bs>
void GhostedBlockVecComm<bs>::setVec(Vec vect)
{
	vv = vect;
}

template <int bs>
void GhostedBlockVecComm<bs>::setModes(const VecCommMode vcm, const InsertMode inm)
{
	cmode = vcm;
	insmode = inm;
}

template <int bs>
void GhostedBlockVecComm<bs>::vecUpdateBegin()
{
	if(commUnderWay)
		throw InvalidCommError("vec update already under way!");

	const ConstGhostedVecHandler<PetscReal> rch(vv);
	const amat::Array2dView<PetscReal> rc(rch.getArray(), msh->gnelem()+msh->gnConnFace(), bs);
	PetscReal *const leftrc = ltv.getLocalArrayLeft();

	if(cmode == DOMAIN_TO_GHOST)
	{
		for(fint icface = msh->gConnBFaceStart(); icface < msh->gConnBFaceEnd(); icface++)
		{
			const fint lelem = msh->gintfac(icface,0);
			for(int ivar = 0; ivar < bs; ivar++)
				leftrc[icface*NDIM+ivar] = rc(lelem,ivar);
		}
	}
	else if(cmode == GHOST_TO_DOMAIN)
	{
		for(fint icface = msh->gConnBFaceStart(); icface < msh->gConnBFaceEnd(); icface++)
		{
			const fint relem = msh->gintfac(icface,1);
			for(int ivar = 0; ivar < bs; ivar++)
				leftrc[icface*NDIM+ivar] = rc(relem,ivar);
		}
	}
	else
		throw UnsupportedOptionError("Invalid communication mode(s)!");
	ltv.updateSharedFacesBegin();
	commUnderWay = true;
}

template <int bs>
void GhostedBlockVecComm<bs>::vecUpdateEnd()
{
	if(!commUnderWay)
		throw InvalidCommError("No vec update under way!");

	ltv.updateSharedFacesEnd();
	commUnderWay = false;

	MutableGhostedVecHandler<PetscReal> rch(vv);
	amat::Array2dMutableView<PetscReal> rc(rch.getArray(), msh->gnelem()+msh->gnConnFace(), bs);
	const PetscReal *const rightrc = ltv.getLocalArrayRight();

	if(cmode == DOMAIN_TO_GHOST && insmode == INSERT_VALUES) {
		for(fint icface = msh->gConnBFaceStart(); icface < msh->gConnBFaceEnd(); icface++)
		{
			const fint relem = msh->gintfac(icface,1);
			for(int idim = 0; idim < bs; idim++)
				rc(relem,idim) = rightrc[icface*NDIM+idim];
		}
	}
	else if(cmode == DOMAIN_TO_GHOST && insmode == ADD_VALUES) {
		for(fint icface = msh->gConnBFaceStart(); icface < msh->gConnBFaceEnd(); icface++)
		{
			const fint relem = msh->gintfac(icface,1);
			for(int idim = 0; idim < bs; idim++)
				rc(relem,idim) += rightrc[icface*NDIM+idim];
		}
	}
	else if(cmode == GHOST_TO_DOMAIN && insmode == ADD_VALUES) {
		for(fint icface = msh->gConnBFaceStart(); icface < msh->gConnBFaceEnd(); icface++)
		{
			const fint lelem = msh->gintfac(icface,0);
			for(int idim = 0; idim < bs; idim++)
				rc(lelem,idim) += rightrc[icface*NDIM+idim];
		}
	}
	else if(cmode == GHOST_TO_DOMAIN && insmode == INSERT_VALUES) {
		for(fint icface = msh->gConnBFaceStart(); icface < msh->gConnBFaceEnd(); icface++)
		{
			const fint lelem = msh->gintfac(icface,0);
			for(int idim = 0; idim < bs; idim++)
				rc(lelem,idim) = rightrc[icface*NDIM+idim];
		}
	}
	else
		throw UnsupportedOptionError("Invalid communication mode(s)!");
}

template class GhostedBlockVecComm<1>;
template class GhostedBlockVecComm<NDIM>;
template class GhostedBlockVecComm<NVARS>;

}
