/** \file
 * \brief Native implementations of and interfaces to some mesh orderings
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

#include <vector>
#include <utility>
#include <set>
#include "utilities/adolcutils.hpp"
#include "utilities/aerrorhandling.hpp"
#include "meshordering.hpp"
#include "details_lineordering.hpp"
#include "ameshutils.hpp"

namespace fvens {

using amat::Array2d;

template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m, const a_real threshold)
{
	const LineConfig lc = findLines(m, threshold);

	// Create the permutation vector using the lines
	std::vector<a_int> ordering(m.gnelem());
	a_int k = 0;
	for(size_t iline = 0; iline < lc.lines.size(); iline++)
	{
		for(a_int i = 0; i < static_cast<a_int>(lc.lines[iline].size()); i++) {
			ordering[k] = lc.lines[iline][i];
			k++;
		}
	}

	for(a_int iel = 0; iel < m.gnelem(); iel++)
		if(lc.celline[iel] == -1)
		{
			ordering[k] = iel;
			k++;
		}

	m.reorder_cells(&ordering[0]);
}

template <typename scalar>
void hybridLineReorder(UMesh2dh<scalar>& m, const a_real threshold, const char *const ordering)
{
	const std::vector<a_int> cellordering = getHybridLineOrdering(m, threshold, ordering);

	// Reorder the mesh
	m.reorder_cells(&cellordering[0]);
}

template <typename scalar>
std::vector<a_int> getHybridLineOrdering(const UMesh2dh<scalar>& m, const a_real threshold,
                                         const char *const ordering)
{
	const LineConfig lc = findLines(m, threshold);
	const GraphVertices gv = createLinePointGraphVertices(m, lc);

	// for(size_t i = 0; i < gv.gverts.size(); i++)
	// 	printf(" %d,%c ", gv.gverts[i].idx, gv.gverts[i].isline ? 'l':'p');
	// printf("\n");

	// printf(" Point list:\n");
	// for(size_t i = 0; i < gv.pointList.size(); i++)
	// 	printf(" %d ", gv.pointList[i]+1+m.gnbface());
	// printf("\n");

	Mat G;
	createLinePointGraph(m, gv, &G);
	// MatView(G, PETSC_VIEWER_STDOUT_SELF);

	// use PETSc to reorder the graph
	const std::vector<PetscInt> grordering = getPetscOrdering(G, ordering);

	// printf("graph ordering:\n");
	// for(size_t i = 0; i < grordering.size(); i++)
	// 	printf(" %d ", grordering[i]+1);
	// printf("\n"); fflush(stdout);

	int ierr = MatDestroy(&G);
	petsc_throw(ierr, "Could not destroy graph matrix!");

	assert(grordering.size() == gv.gverts.size());

	std::vector<GraphVertex> rogv(gv.gverts.size());

	// Reorder the graph of lines and points. This way of ordering matches UMesh2dh::reorder_cells.
	for(size_t i = 0; i < rogv.size(); i++)
		rogv[i] = gv.gverts[grordering[i]];

	std::vector<a_int> cellordering(m.gnelem());

	// Build ordering of all cells
	a_int iord=0;
	for(a_int i = 0; i < static_cast<a_int>(rogv.size()); i++)
	{
		if(rogv[i].isline) {
			const int linesize = static_cast<int>(lc.lines[rogv[i].idx].size());
			for(int icell = 0; icell < linesize; icell++)
			{
				cellordering[iord] = lc.lines[rogv[i].idx][icell];
				iord++;
			}
		}
		else {
			cellordering[iord] = gv.pointList[rogv[i].idx];
			iord++;
		}
	}

	assert(iord == m.gnelem());
	return cellordering;
}

struct LocalAnisotropies
{
	/// Measure of local anisotropy for each cell for each neighbor, ordered decreasing
	Array2d<a_real> aniso;
	/// Local face index ordered according to \ref aniso
	Array2d<EIndex> faceIdx;
	/// Number of real neighbors for each cell
	std::vector<int> nRealNbrs;
};

/// Returns the edge weights about each cell, ordered by decreasing weight
template <typename scalar>
LocalAnisotropies computeWeights(const UMesh2dh<scalar>& m)
{
	LocalAnisotropies la;
	la.aniso.resize(m.gnelem(), m.gmaxnfael());
	la.faceIdx.resize(m.gnelem(), m.gmaxnfael());
	for(a_int i = 0; i < m.gnelem(); i++)
		for(int j = 0; j < m.gmaxnfael(); j++)
			la.faceIdx(i,j) = -1;
	la.nRealNbrs.resize(m.gnelem());

	Array2d<scalar> ccentres(m.gnelem(),NDIM);
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		kernel_computeCellCentreAoS(m, iel, &ccentres(iel,0));
	}

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m.gnelem(); iel++)
	{
		std::vector<std::pair<a_real,EIndex>> elaniso;
		elaniso.reserve(m.gnfael(iel));
		a_real minw = 1e20;

		for(EIndex j = 0; j < m.gnfael(iel); j++)
		{
			const a_int jel = m.gesuel(iel,j);
			// Skip ghost neighbors across boundary faces
			if(jel >= m.gnelem())
				continue;

			std::pair<a_real,EIndex> nbrwt;

			a_real dist = 0;
			for(int idim = 0; idim < NDIM; idim++)
				dist += std::pow(getvalue(ccentres(iel,idim)-ccentres(jel,idim)),2);
			nbrwt.first = 1.0/std::sqrt(dist);

			if(nbrwt.first < minw)
				minw = nbrwt.first;

			nbrwt.second = j;
			elaniso.push_back(nbrwt);
		}

		for(size_t j = 0; j < elaniso.size(); j++)
			elaniso[j].first /= minw;

		// sort by *decreasing* weight
		std::sort(elaniso.begin(), elaniso.end(),
		          [](std::pair<a_real,EIndex> a, std::pair<a_real,EIndex> b) {
			          return a.first > b.first;
		          });

		la.nRealNbrs[iel] = static_cast<int>(elaniso.size());
		for(EIndex j = 0; j < la.nRealNbrs[iel]; j++) {
			la.aniso(iel,j) = elaniso[j].first;
			la.faceIdx(iel,j) = elaniso[j].second;
		}
	}

	return la;
}

template <typename scalar>
LineConfig findLines(const UMesh2dh<scalar>& m, const a_real threshold)
{
	LineConfig lc;
	const LocalAnisotropies la = computeWeights(m);

	lc.celline.assign(m.gnelem(), -1);

	// Try to build a line starting at each boundary cell
	for(a_int iface = m.gPhyBFaceStart(); iface < m.gPhyBFaceEnd(); iface++)
	{
		std::vector<a_int> linelems;
		const a_int belem = m.gintfac(iface,0);
		if(lc.celline[belem] >= 0) {
#ifdef DEBUG
			printf("  lineReorder: A boundary cell is already part of a line.\n");
			fflush(stdout);
#endif
			continue;
		}

		bool endoftheline = false;
		a_int curelem = belem;

		while(!endoftheline)
		{
			if(la.aniso(curelem,0) > threshold) {
				linelems.push_back(curelem);
				lc.celline[curelem] = static_cast<int>(lc.lines.size());
			}
			else
				break;

			endoftheline = true;

			for(EIndex j = 0; j < la.nRealNbrs[curelem]; j++)
			{
				const a_int nbrelem = m.gesuel(curelem, la.faceIdx(curelem,j));

				if(lc.celline[nbrelem]==-1 && la.aniso(curelem,j) > threshold) {
					curelem = nbrelem;
					endoftheline = false;
					break;
				}
			}
		}

		if(linelems.size() > 1)
			lc.lines.push_back(linelems);
		else if(linelems.size() == 1) {
			lc.celline[linelems[0]] = -1;
		}
	}

	printf(" lineReorder: Found %ld lines.\n", lc.lines.size());

	return lc;
}

template <typename scalar>
GraphVertices createLinePointGraphVertices(const UMesh2dh<scalar>& m, const LineConfig& lc)
{
	assert(lc.celline.size() == static_cast<size_t>(m.gnelem()));

	GraphVertices gv;

	const a_int nlines = static_cast<a_int>(lc.lines.size());

	gv.cellsToPtsMap.assign(m.gnelem(), -1);
	gv.pointList.reserve(m.gnelem());
	for(size_t i = 0; i < lc.celline.size(); i++)
		if(lc.celline[i] == -1) {
			gv.pointList.push_back(i);
			gv.cellsToPtsMap[i] = static_cast<a_int>(gv.pointList.size()-1);
		}
	gv.pointList.shrink_to_fit();

	const a_int npoints = static_cast<a_int>(gv.pointList.size());

	gv.gverts.resize(nlines+npoints);

	for(a_int iline = 0; iline < nlines; iline++) {
		gv.gverts[iline].isline = true;
		gv.gverts[iline].idx = iline;
	}
	for(a_int ipoin = 0; ipoin < npoints; ipoin++) {
		gv.gverts[nlines+ipoin].isline = false;
		gv.gverts[nlines+ipoin].idx = ipoin;
	}

	gv.lc = &lc;
	return gv;
}

template <typename scalar>
void createLinePointGraph(const UMesh2dh<scalar>& m, const GraphVertices& gv, Mat *const G)
{
	const a_int nlines = static_cast<a_int>(gv.lc->lines.size());
	const a_int npoints = static_cast<a_int>(gv.pointList.size());
	// printf("createLPGraph: Num lines = %d\n", nlines);
	// printf("createLPGraph: Num points = %d\n", npoints);

	// Set up Petsc mat
	int ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nlines+npoints, nlines+npoints, 4, NULL, G);
	petsc_throw(ierr, "Could not create line-block graph matrix!");

	ierr = MatSetOption(*G, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
	petsc_throw(ierr, "Could not set Mat option");
	
	// find connections between lines and other lines or points
	for(a_int iline = 0; iline < nlines; iline++)
	{
		std::set<a_int> linenbrs, pointnbrs;

		for(int icell = 0; icell < static_cast<int>(gv.lc->lines[iline].size()); icell++)
		{
			const a_int cell = gv.lc->lines[iline][icell];
			for(EIndex j = 0; j < m.gnfael(cell); j++)
			{
				const a_int cellnbr = m.gesuel(cell,j);
				if(cellnbr >= m.gnelem())
					continue;

				if(gv.lc->celline[cellnbr] >= 0) {
					linenbrs.insert(gv.lc->celline[cellnbr]);
				}
				else {
					assert(gv.cellsToPtsMap[cellnbr] >= 0);
					pointnbrs.insert(gv.cellsToPtsMap[cellnbr]+nlines);
				}
			}
		}

		// add to graph
		const a_real val = 1.0;
#pragma omp critical
		{
			ierr = MatSetValues(*G, 1, &iline, 1, &iline, &val, INSERT_VALUES);
		}
		for( a_int nbr : linenbrs ) {
#pragma omp critical
			{
				ierr = MatSetValues(*G, 1, &iline, 1, &nbr, &val, INSERT_VALUES);
			}
		}
		for(a_int nbr : pointnbrs) {
#pragma omp critical
			{
				ierr = MatSetValues(*G, 1, &iline, 1, &nbr, &val, INSERT_VALUES);
			}
		}
	}

	petsc_throw(ierr, "Error in setting values of line-point graph!");

	// find connections between each point and neighbouring lines or points and add them to the graph
	for(a_int ipoin = 0; ipoin < npoints; ipoin++)
	{
		const a_int cell = gv.pointList[ipoin];
		const a_int ipdx = ipoin+nlines;
		const a_real val = 1.0;

#pragma omp critical
		{
			ierr = MatSetValues(*G, 1, &ipdx, 1, &ipdx, &val, INSERT_VALUES);
		}

		for(EIndex j = 0; j < m.gnfael(cell); j++)
		{
			const a_int cellnbr = m.gesuel(cell,j);
			if(cellnbr >= m.gnelem())
				continue;

			a_int nbidx;
			if(gv.lc->celline[cellnbr] >= 0)
			{
				nbidx = gv.lc->celline[cellnbr];
			}
			else
			{
				assert(gv.cellsToPtsMap[cellnbr] >= 0);
				nbidx = gv.cellsToPtsMap[cellnbr]+nlines;
			}

#pragma omp critical
			{
				ierr = MatSetValues(*G, 1, &ipdx, 1, &nbidx, &val, INSERT_VALUES);
			}
		}
	}

	ierr = MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY);
	ierr = MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY);

	petsc_throw(ierr, "Assembly of line-point graph failed!");
}

std::vector<PetscInt> getPetscOrdering(Mat G, const char *const ordering)
{
	PetscInt rows, cols;
	int ierr = MatGetSize(G, &rows, &cols); petsc_throw(ierr, "Could not get mat size!");
	assert(rows == cols);

	if(!strcmp(ordering,"natural")) {
		printf(" No further reordering of the graph to be done.\n");
		std::vector<PetscInt> ord(rows);
		for(PetscInt i = 0; i < rows; i++)
			ord[i] = i;
		return ord;
	}

	printf(" Further reordering the graph in %s ordering.\n", ordering);

	IS rperm, cperm;
	const PetscInt *rinds, *cinds;
	ierr = MatGetOrdering(G, ordering, &rperm, &cperm); petsc_throw(ierr, "Could not get ordering!");
	ierr = ISGetIndices(rperm, &rinds);
	ierr = ISGetIndices(cperm, &cinds);

	// check for symmetric permutation
	for(a_int i = 0; i < rows; i++)
		assert(rinds[i] == cinds[i]);

	std::vector<PetscInt> ord(rows);
	for(PetscInt i = 0; i < rows; i++)
		ord[i] = rinds[i];

	ierr = ISRestoreIndices(rperm, &rinds);
	ierr = ISDestroy(&rperm);
	ierr = ISDestroy(&cperm);
	petsc_throw(ierr, "Could not destroy IS");

	return ord;
}

template void lineReorder(UMesh2dh<a_real>& m, const a_real threshold);
template std::vector<a_int> getHybridLineOrdering(const UMesh2dh<a_real>& m, const a_real threshold,
                                                  const char *const ordering);
template void hybridLineReorder(UMesh2dh<a_real>& m, const a_real threshold, const char *const ordering);

}
