/** \file
 * \brief Routines for distributing a mesh among available processes
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MESH_PARTITIONING_H
#define FVENS_MESH_PARTITIONING_H

#include "amesh2dh.hpp"

namespace fvens {

StatusCode partitionMesh(UMesh2dh<a_real>& mesh);

}

#endif
