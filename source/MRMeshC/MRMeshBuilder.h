#pragma once
#include "MRMeshFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// the function finds groups of mesh vertices located closer to each other than \param closeDist, and unites such vertices in one;
/// then the mesh is rebuilt from the remaining triangles
/// \param optionalVertOldToNew is the mapping of vertices: before -> after
/// \param uniteOnlyBd if true then only boundary vertices can be united, all internal vertices (even close ones) will remain
/// \return the number of vertices united, 0 means no change in the mesh
MRMESHC_API int mrMeshBuilderUniteCloseVertices( MRMesh* mesh, float closeDist, bool uniteOnlyBd, MRVertMap* optionalVertOldToNew );

MR_EXTERN_C_END
